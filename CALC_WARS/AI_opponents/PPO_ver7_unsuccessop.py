import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, Any, Tuple
import random
from collections import deque
from Game_env_PPO import Game, CardType, OperatorType, Card
from typing import List, Dict, Tuple, Optional 


class PPOTransformer(nn.Module):
    """混合架构: MLP处理静态特征 + Transformer处理序列"""
    def __init__(self, obs_dim=131, action_dim=29, hidden_size=512, 
                 seq_len=9,trace_len= 10, d_model=128, nhead=4, num_layers=2):
        super().__init__()

        self.seq_len = seq_len
        self.trace_len = trace_len
        self.d_model = d_model
        
        # 1. 静态特征编码器（输入79维）
        self.static_encoder = nn.Sequential(
            nn.Linear(79, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # 新增：数值编码器（输入2维：value, depth）
        self.symbolic_encoder = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # 新增：Trace序列编码器（3维输入：value, depth, validity）
        self.trace_encoder = nn.Sequential(
            nn.Linear(3, d_model),  # 输入维度改为3
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Trace位置编码（与Card序列独立）
        self.trace_pos_encoding = nn.Parameter(torch.zeros(1, self.trace_len, d_model))

        # 新增：Cross-Attention层（Trace查询Card）
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=0.2,
            batch_first=True
        )
                
        # 2. 序列特征编码器
        self.card_index_embedding = nn.Embedding(21, d_model, padding_idx=20)
        self.card_type_embedding = nn.Embedding(5, d_model, padding_idx=4)
        self.pos_encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=hidden_size,
            dropout=0.2,
            batch_first=True,
            activation='gelu'
        )
        self.card_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.trace_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 注意力池化层
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
        
        # 3. 特征融合层（输入维度修正）
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + d_model * 2, hidden_size),  # 注意：这里只需要+d_model，因为seq_pooled已经是d_model
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 新增：Trace的注意力池化层（独立参数）
        self.trace_attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

        # 新增中间预测头
        self.value_head_mid = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.value_head_start = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 4. 策略和价值头
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.Parameter):
                nn.init.normal_(module, mean=0, std=0.02)

        for transformer in [self.card_transformer, self.trace_transformer]:
            for layer in transformer.layers:
                # 使用正交初始化
                if hasattr(layer.self_attn, 'in_proj_weight'):
                    nn.init.orthogonal_(layer.self_attn.in_proj_weight, gain=0.5)
                    nn.init.constant_(layer.self_attn.in_proj_bias, 0)

        # 符号编码器使用小gain初始化
        for module in self.symbolic_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.5)
                nn.init.constant_(module.bias, 0)

        # 初始化Trace编码器
        for module in self.trace_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.5)
                nn.init.constant_(module.bias, 0)

        # 初始化Cross-Attention
        nn.init.normal_(self.cross_attention.in_proj_weight, mean=0, std=0.02)
        nn.init.constant_(self.cross_attention.in_proj_bias, 0)
        nn.init.normal_(self.cross_attention.out_proj.weight, mean=0, std=0.02)
        nn.init.constant_(self.cross_attention.out_proj.bias, 0)

    def get_features(self, x: torch.Tensor):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        static_features = x[:, :79]
        sequence_compressed = x[:, 79:97]
        
        # 拆分序列
        seq_tokens = sequence_compressed.reshape(-1, self.seq_len, 2)
        card_indices = (seq_tokens[:, :, 0] * 20).long().clamp(0, 20)  # 关键修复
        card_types = seq_tokens[:, :, 1].float()
        
        # 合并为2D输入
        card_metadata = torch.stack([
            card_indices.float(), 
            card_types,
        ], dim=-1)  # shape: [batch, seq_len, 2]

        # 编码
        static_encoded = self.static_encoder(static_features)
        index_emb = self.card_index_embedding(card_indices)
        metadata_emb = self.symbolic_encoder(card_metadata)
        
        # 融合
        card_seq_emb = index_emb + metadata_emb + self.pos_encoding

        # 3. Trace序列（新增）
        trace_raw = x[:, 97:127].reshape(-1, self.trace_len, 3)  # 原117改为127，2改为3
        trace_values = trace_raw[:, :, 0]
        trace_depths = trace_raw[:, :, 1]
        trace_validity_raw = trace_raw[:, :, 2]
    
        validity_mask = (trace_validity_raw != -2.0).float()  # PAD标记为-2.0
        trace_features = torch.stack([
            trace_values,
            trace_depths,
            validity_mask
        ], dim=-1)  # [batch, seq_len, 3]
        
        trace_emb = self.trace_encoder(trace_features) + self.trace_pos_encoding
            
        # Padding mask
        card_padding_mask = (card_indices == 20) | (card_metadata[..., 0] == 4)
        trace_padding_mask = (validity_mask == 0)  # validity=0是PAD

        # 5. Cross-Attention：Trace查询Card
        #    这样Trace的每一步都能关注导致它的Card
        trace_attended, _ = self.cross_attention(
            query=trace_emb,      # [batch, seq_len, d_model]
            key=card_seq_emb,     # [batch, seq_len, d_model]  
            value=card_seq_emb,
            key_padding_mask=card_padding_mask,
            need_weights=False
        )

        # 6. 独立Transformer编码（双塔结构）
        card_encoded = self.card_transformer(card_seq_emb, src_key_padding_mask=card_padding_mask)
        trace_encoded = self.trace_transformer(trace_attended, src_key_padding_mask=trace_padding_mask)

        # 7. 独立注意力池化
        card_pooled = self._attention_pool(card_encoded, card_padding_mask, mode='card')
        trace_pooled = self._attention_pool(trace_encoded, trace_padding_mask, mode='trace')

        # 最终融合
        combined = torch.cat([static_encoded, card_pooled, trace_pooled], dim=-1)
        features = self.fusion(combined)
        
        return features
    
    def _attention_pool(self, seq: torch.Tensor, mask: torch.Tensor, mode: str):
        if mode == 'card':
            scores = self.attention_pool(seq)
        else:  # trace
            scores = self.trace_attention_pool(seq)
        
        # 处理mask
        scores = scores.masked_fill(mask.unsqueeze(-1), -1e4)
        weights = F.softmax(scores, dim=1)
        return torch.sum(seq * weights, dim=1)
    
    def forward(self, x):
        features = self.get_features(x)
        logits = self.policy_head(features)
        values = self.value_head(features)
        return logits, values

class PPORolloutCollector:
    """修正版：使用统一的数据类型，避免混合list和numpy"""
    def __init__(self, buffer_size: int = 2048):
        self.buffer_size = buffer_size
        self.reset()
    
    def reset(self):
        # 统一使用列表存储，便于动态扩展
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []      # 这里保持为list
        self.rewards = []
        self.dones = []
        self.masks = []
        self.ptr = 0
    
    def add(self, state, action, log_prob, value, reward, done, mask):
        """单条添加"""
        if self.ptr >= self.buffer_size:
            raise RuntimeError("Buffer overflow!")
            
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.masks.append(mask)
        self.ptr += 1

    def add_batch(self, states, actions, log_probs, values, rewards, dones, masks):
        """批量添加经验，如果缓冲区空间不足则跳过该批次"""
        batch_size = len(states)
        
        # 检查是否有足够空间
        if self.ptr + batch_size > self.buffer_size:
            # 空间不足，跳过这批数据（不抛出错误）
            # 可以打印警告以便调试（可选）
            if self.ptr < self.buffer_size:  # 只在非满状态下打印
                print(f"Warning: Skipping batch of size {batch_size} "
                    f"(buffer has {self.buffer_size} slots, {self.ptr} filled)")
            return
        
        # 有足够空间，正常添加
        self.states.extend(states)
        self.actions.extend(actions)
        self.log_probs.extend(log_probs)
        self.values.extend(values)
        self.rewards.extend(rewards)
        self.dones.extend(dones)
        self.masks.extend(masks)
        
        self.ptr += batch_size
    
    def finalize(self, last_value: float = 0.0):
        """在episode结束时调用，values比rewards多1个用于GAE"""
        self.values.append(last_value)  # 现在self.values是list，append可用
    
    def is_full(self) -> bool:
        return self.ptr >= self.buffer_size
    
    def get_batch(self) -> Dict[str, np.ndarray]:
        """返回用于PPO训练的numpy数组（在训练前统一转换）"""
        # 转换为numpy数组并归一化优势值等操作
        return {
            'states': np.array(self.states, dtype=np.float32),
            'actions': np.array(self.actions, dtype=np.int64),
            'log_probs': np.array(self.log_probs, dtype=np.float32),
            'values': np.array(self.values, dtype=np.float32),  # 这里转换为numpy
            'rewards': np.array(self.rewards, dtype=np.float32),
            'dones': np.array(self.dones, dtype=np.float32),
            'masks': np.array(self.masks, dtype=bool),
        }


class PPOTrainer:
    """标准PPO训练器:支持GAE + Value Clipping"""
    def __init__(self, 
                 policy_net: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scaler: torch.cuda.amp.GradScaler,
                 config: Dict[str, Any]):
    
        self.config = config
        self.policy_net = policy_net
        self.optimizer = optimizer
        self.scaler = scaler  
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.mini_batch_size = config.get('mini_batch_size', 64)
        self.clip_param = config.get('clip_param', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.gamma = config.get('gamma', 0.99)
        self.lam = config.get('lam', 0.95)
        self.reward_scaling = config.get('reward_scaling', 50.0)
    
    def compute_gae(self, rewards: np.ndarray, values: np.ndarray, 
               dones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        rewards = rewards / self.reward_scaling
        
        last_gae = 0
        # values长度是len(rewards)+1，values[-1] = V(s_{T+1})
        last_value = values[-1]  # 初始V(s_{T+1})
        
        for t in reversed(range(len(rewards))):
            # 如果done_t=1，则V(s_{t+1})=0，否则使用计算的值
            next_value = last_value * (1 - dones[t])
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae
            returns[t] = advantages[t] + values[t]
            
            last_gae = advantages[t]
            last_value = values[t]  # 保存当前V(s_t)作为下次迭代的V(s_{t+1})
        
        # 标准化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # returns通常不标准化，或仅用均值平移
        
        return advantages, returns
    
    def update(self, rollout: Dict[str, np.ndarray]) -> float:
        """执行PPO更新 支持mini-batch"""
        device = next(self.policy_net.parameters()).device
        
        # 计算GAE（在整个rollout上）
        advantages, returns = self.compute_gae(
            rollout['rewards'], rollout['values'], rollout['dones']
        )
        
        # 转换为Tensor
        states = torch.FloatTensor(rollout['states']).to(device)
        actions = torch.LongTensor(rollout['actions']).to(device)
        old_log_probs = torch.FloatTensor(rollout['log_probs']).to(device)
        old_values = torch.FloatTensor(rollout['values']).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        returns = torch.FloatTensor(returns).to(device)
        masks = torch.BoolTensor(rollout['masks']).to(device)
        
        total_losses = []
        
        # PPO Epochs
        for _ in range(self.ppo_epochs):
            # 生成随机索引
            indices = torch.randperm(len(states))
            
            # Mini-batch更新
            for start in range(0, len(states), self.mini_batch_size):
                end = min(start + self.mini_batch_size, len(states))
                if end - start < self.mini_batch_size // 2:  # 太小就跳过
                    continue

                batch_idx = indices[start:end]
                
                # Mini-batch数据
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_masks = masks[batch_idx]
                batch_old_values = old_values[batch_idx]
                
                with torch.cuda.amp.autocast():  # 新增：自动混合精度上下文
                    logits, values = self.policy_net(batch_states)
                    logits = logits.masked_fill(~batch_masks, -1e4)
                    
                    # 策略损失（原有代码不变）
                    dist = Categorical(logits=logits)
                    new_log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # 价值损失（原有代码不变）
                    values_pred_clipped = batch_old_values + torch.clamp(
                        values.squeeze(-1) - batch_old_values, -self.clip_param, self.clip_param
                    )
                    value_loss1 = F.mse_loss(values.squeeze(-1), batch_returns, reduction='none')
                    value_loss2 = F.mse_loss(values_pred_clipped, batch_returns, reduction='none')
                    value_loss = torch.max(value_loss1, value_loss2).mean()

                    total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # 修改反向传播逻辑
                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()  # 修改：用scaler缩放loss
                self.scaler.unscale_(self.optimizer)      # 新增：先unscale再裁剪
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)          # 修改：用scaler更新
                self.scaler.update()                      # 新增：更新scaler
                
                total_losses.append(total_loss.item())
        
        return np.mean(total_losses)


class SelfPlayAgent:
    def __init__(self, policy_net: nn.Module, device: torch.device):
        self.policy_net = policy_net
        self.device = device
    
    # 保持原有单条接口（用于预训练和调试）
    def get_action(self, state: np.ndarray, action_mask: np.ndarray, 
                   player_id:int = 0, deterministic: bool = False) -> Tuple[int, float, float]:
        # 复用批量逻辑
        states = state[np.newaxis, ...]  # [1, 131]
        masks = action_mask[np.newaxis, ...]  # [1, 29]
        pids = [player_id]
        
        actions, log_probs, values = self.get_actions_batch(
            states, masks, pids, deterministic
        )
        return actions[0], log_probs[0], values[0]
        
    # 新增批量方法
    def get_actions_batch(self, 
                          states: np.ndarray, 
                          masks: np.ndarray,
                          player_ids: List[int], 
                          deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        批量推理入口
        Args:
            states: 状态数组 [batch_size, 131]
            masks: mask数组 [batch_size, 29]
            player_ids: 玩家ID列表 [batch_size]
            deterministic: 是否确定性策略
        Returns:
            actions: [batch_size]
            log_probs: [batch_size]
            values: [batch_size]
        """
        batch_size = states.shape[0]
        
        # 1. 转换为tensor并移至GPU（一次操作）
        states_t = torch.from_numpy(states).to(self.device, non_blocking=True)
        masks_t = torch.from_numpy(masks).bool().to(self.device, non_blocking=True)
        
        with torch.no_grad():
            # 2. 单次前向传播（核心优化点）
            logits, values = self.policy_net(states_t)  # [batch, 29], [batch, 1]
            
            # 3. 批量mask处理（关键优化）
            # 找到每个环境的最小合法logit值
            min_valid_logits = torch.full((batch_size,), -1e4, device=self.device)
            for i in range(batch_size):
                valid_logits = logits[i][masks_t[i]]
                if valid_logits.numel() > 0:
                    min_valid_logits[i] = valid_logits.min() - 50.0
            
            # 应用mask
            logits = torch.where(masks_t, logits, min_valid_logits.unsqueeze(1))
            
            # 4. 噪声添加（对手玩家）
            if not deterministic:
                # 批量生成噪声
                noise = torch.randn_like(logits) * 0.05
                # 只对player_id==1的环境添加噪声
                noise_mask = torch.tensor([pid == 1 for pid in player_ids], 
                                         device=self.device, dtype=torch.bool)
                noise = noise * noise_mask.unsqueeze(1)
                # 屏蔽无效位置
                noise = noise.masked_fill(~masks_t, 0)
                logits = logits + noise
            
            # 5. 批量采样
            dist = Categorical(logits=logits)
            if deterministic:
                actions = logits.argmax(dim=-1)
            else:
                actions = dist.sample()
            
            # 6. 收集结果
            log_probs = dist.log_prob(actions)
            
            return (
                actions.cpu().numpy(),
                log_probs.cpu().numpy(),
                values.squeeze(-1).cpu().numpy()
            )


class TrainingManager:
    """优化的训练管理器"""
    def __init__(self, env, config: Dict[str, Any]):
        self.env = env
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.cuda.amp.GradScaler()  # 混合精度缩放器

        self.pretrain_steps = config.get('pretrain_steps', 300)
        self.pretrain_lr = config.get('pretrain_lr', 1e-5)
        
        # 将trainer引用注入环境（用于symbolic_execution_trace）
        self.env.trainer = self

        # 课程学习配置
        self.use_curriculum = config.get('use_curriculum', False)
        self.current_stage = 0
        self.stage_boundaries = [1200, 4000]  # 阶段切换的episode阈值
        
        # 网络
        self.policy_net = PPOTransformer(
            obs_dim=131,  
            action_dim=29,
            hidden_size=config.get('hidden_size', 512),
            seq_len=9,      # 必须与环境中的最大表达式长度一致
            d_model=config.get('d_model', 128),      
            nhead=config.get('nhead', 4),            
            num_layers=config.get('transformer_layers', 2) 
        ).to(self.device)

        # 更新优化器（Transformer需要更低学习率）
        self.optimizer = optim.AdamW(  # 改用AdamW，权重衰减更稳定
            self.policy_net.parameters(),
            lr=config.get('lr', 5e-5),      # [降低] 从1e-4降至5e-5
            eps=config.get('adam_eps', 1e-5),
            weight_decay=config.get('weight_decay', 0.01)  # [ADD] 权重衰减
        )
        
        # 训练器
        self.trainer = PPOTrainer(
            policy_net=self.policy_net,
            optimizer=self.optimizer,
            scaler=self.scaler,
            config = config
        )
        
        # 智能体
        self.agent = SelfPlayAgent(self.policy_net, self.device)
        
        # 数据收集器
        self.collector = PPORolloutCollector(
            buffer_size=config.get('buffer_size', 2048)
        )
        
        # 训练记录
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.losses = deque(maxlen=100)
        self.episode = 0
        
        # 熵衰减
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.entropy_decay = config.get('entropy_decay', 0.995)
        self.entropy_min = config.get('entropy_min', 0.001)

        self.expr_stats = {
        'valid_exprs': deque(maxlen=1000),  # 有效表达式比例
        'avg_expr_len': deque(maxlen=1000), # 平均长度
        'skill_triggers': deque(maxlen=1000), # 技能触发率
        'bracket_bal': deque(maxlen=1000), #括号平衡
        'bracket_num':deque(maxlen=1000), #总括号数
        'zero_damage_raw': deque(maxlen=1000),  # 原始零伤害率（游戏规则）
        'symbolic_loss': deque(maxlen=100),      # 符号预测损失
        'value_pred_acc': deque(maxlen=100),     # value head预测准确率
        'bracket_effectiveness': deque(maxlen=1000), # 括号有效性乘数
    }
        
    def update_curriculum_stage(self):
        """根据episode数自动升级课程阶段"""
        if not self.use_curriculum:
            return
        
        # 确定当前episode对应的stage
        if self.episode < self.stage_boundaries[0]:
            new_stage = 0
        elif self.episode < self.stage_boundaries[1]:
            new_stage = 1
        else:
            new_stage = 2
        
        # 仅在升级时执行更新
        if new_stage > self.current_stage:
            print(f"\n{'='*55}")
            print(f"  课程学习升级: Stage {self.current_stage} → Stage {new_stage}")
            print(f"  当前Episode: {self.episode}")
            print(f"{'='*55}\n")
            
            self.current_stage = new_stage
            self.env.set_curriculum_stage(new_stage)  # 更新环境

    
    def collect_rollout(self) -> Tuple[float, int]:
        """收集一个episode, 添加到collector"""
        state = self.env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done and episode_length < self.config.get('max_episode_steps', 512):
            # 当前玩家
            current_pid = 0 if self.env.current_player == self.env.player1 else 1
            
            # 获取动作
            action_mask = self.env.get_valid_actions_mask()
            action, log_prob, value = self.agent.get_action(state, action_mask, current_pid)

            if action == 26:
                # 在step()之前记录当前状态
                expr_length = len(self.env.selected_cards)
                result = self.env.calculate_expression(self.env.selected_cards)
                is_valid = result is not None and result >= 0
                raw_damage = self.env.target.get_damage(result)

                handmaded = []
                for card in self.env.selected_cards:
                    if card.card_type == CardType.OPERATOR:
                        if card.operator_type == OperatorType.LEFTBRA:
                            handmaded.append(1)
                        elif card.operator_type == OperatorType.RIGHTBRA:
                            handmaded.append(-1)
                bracket_balance = sum(handmaded) if handmaded else 0
                bracket_count = len(handmaded)

                if bracket_count > 0 and random.random() < 0.0005:
                    print(f"[DEBUG] Expr: {[str(c) for c in self.env.selected_cards]}, "
                    f"Result: {result}, Valid: {result is not None}")

                if random.random() < 0.0001:  # 0.1%概率输出符号trace
                    trace = self.env.symbolic_executor.get_trace()
                    print(f"[SYMBOLIC] Trace: {[f'v={v:.1f},d={d}' for _,v,d,_ in trace]}")

                if random.random() < 0.0005:
                    print(f"[DEBUG] Expr: {[str(c) for c in self.env.selected_cards]}, "
                    f"Result: {result}, Valid: {result is not None}")
                
                # 执行动作
                next_state, reward, done, info = self.env.step(action)
                
                # 记录统计（使用step()之前的数据）
                self.expr_stats['avg_expr_len'].append(expr_length)
                self.expr_stats['valid_exprs'].append(float(is_valid))
                self.expr_stats['bracket_bal'].append(bracket_balance)
                self.expr_stats['bracket_num'].append(bracket_count)
                self.expr_stats['zero_damage_raw'].append(1.0 if raw_damage == 0 else 0.0)
                if is_valid:
                    skill_triggered = 1.0 if info.get('skill_triggered') else 0.0
                    self.expr_stats['skill_triggers'].append(skill_triggered)

            else:
                next_state, reward, done, _ = self.env.step(action)
            
            # 存储（对齐PPO论文的标志位：reward放在当前状态后）
            self.collector.add(
                state=state,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=reward,
                done=done,
                mask=action_mask
            )
            
            state = next_state
            episode_reward += reward
            episode_length += 1

        
        # 在episode结束时调用finalize
        if done:
            # 正常结束，V(s_final) = 0
            self.collector.finalize(last_value=0.0)
        else:
            # 超时截断，需要计算V(s_{t+1})
            _, _, last_value = self.agent.get_action(state, self.env.get_valid_actions_mask())
            self.collector.finalize(last_value=last_value)
        
        return episode_reward, episode_length
    
    def collect_rollout_batch(self, num_envs: int = 8) -> Tuple[float, int]:
        """
        批量收集rollout - 修复索引错误版本
        """
        # 1. 创建环境列表
        envs = [Game(self.config.get('use_curriculum', False)) for _ in range(num_envs)]
        for env in envs:
            if self.use_curriculum:
                env.set_curriculum_stage(self.current_stage)
        
        # 2. 初始化状态（所有环境）
        states = np.stack([env.reset() for env in envs], axis=0)
        masks = np.stack([env.get_valid_actions_mask() for env in envs], axis=0)
        player_ids = [0 if env.current_player == env.player1 else 1 for env in envs]
        
        # 3. 记录每个环境的统计信息
        episode_rewards = np.zeros(num_envs)
        episode_lengths = np.zeros(num_envs, dtype=np.int32)
        
        # 4. 主循环：使用 while 而不是 for，避免迭代器失效
        while len(envs) > 0 and not self.collector.is_full():
            # 为当前所有环境批量推理
            actions, log_probs, values = self.agent.get_actions_batch(
                states, masks, player_ids, deterministic=False
            )
            
            # 5. 准备下一轮的数据容器（新建列表，不修改原列表）
            next_states = []
            next_masks = []
            next_player_ids = []
            next_rewards = []
            next_lengths = []
            next_envs = []  # 新的活跃环境列表
            
            # 6. 遍历当前环境（现在安全，因为不修改envs列表）
            for i, env in enumerate(envs):
                # 执行动作
                next_state, reward, done, info = env.step(actions[i])
                
                # 记录统计
                episode_rewards[i] += reward
                episode_lengths[i] += 1
                
                # 暂存经验到环境对象
                if not hasattr(env, '_temp_buffer'):
                    env._temp_buffer = []
                env._temp_buffer.append({
                    'state': states[i].copy(),  # 复制避免引用问题
                    'action': actions[i],
                    'log_prob': log_probs[i],
                    'value': values[i],
                    'reward': reward,
                    'done': done,
                    'mask': masks[i].copy()
                })
                
                if done:
                    # 环境结束：计算last_value并批量添加到collector
                    _, _, last_value = self.agent.get_action(
                        next_state, env.get_valid_actions_mask(), player_ids[i]
                    )
                    
                    # 批量添加该环境的所有经验
                    batch_data = {
                        'states': np.array([t['state'] for t in env._temp_buffer]),
                        'actions': np.array([t['action'] for t in env._temp_buffer]),
                        'log_probs': np.array([t['log_prob'] for t in env._temp_buffer]),
                        'values': np.array([t['value'] for t in env._temp_buffer]),
                        'rewards': np.array([t['reward'] for t in env._temp_buffer]),
                        'dones': np.array([t['done'] for t in env._temp_buffer]),
                        'masks': np.array([t['mask'] for t in env._temp_buffer]),
                    }
                    self.collector.add_batch(**batch_data)
                    self.collector.finalize(last_value)
                    
                    # 检查collector是否已满（提前退出）
                    if self.collector.is_full():
                        # 清空所有环境buffer
                        for e in envs:
                            if hasattr(e, '_temp_buffer'):
                                delattr(e, '_temp_buffer')
                        envs.clear()
                        break  # 跳出for循环
                    
                    # 清理已结束的环境
                    del env._temp_buffer
                    # **关键：不加入next_envs，相当于移除了**
                else:
                    # 环境继续：加入下一轮
                    next_states.append(next_state)
                    next_masks.append(env.get_valid_actions_mask())
                    next_player_ids.append(0 if env.current_player == env.player1 else 1)
                    next_rewards.append(episode_rewards[i])
                    next_lengths.append(episode_lengths[i])
                    next_envs.append(env)
            
            # 7. 更新下一轮的状态（在for循环外部）
            if envs:  # 如果还有活跃环境
                states = np.stack(next_states, axis=0)
                masks = np.stack(next_masks, axis=0)
                player_ids = next_player_ids
                episode_rewards = np.array(next_rewards)
                episode_lengths = np.array(next_lengths)
                envs = next_envs  # **替换整个列表，而不是修改原列表**
            
            # 如果collector已满，退出while循环
            if self.collector.is_full():
                break
        
        # 8. 返回统计信息（只统计完成的episode）
        return episode_rewards.mean(), episode_lengths.mean()
    

    def pretrain_symbolic(self):
        """阶段1：符号自监督预训练"""
        print("=== 启动符号预训练 ===")
        
        # 冻结策略头，只训练Transformer和符号编码器
        for param in self.policy_net.policy_head.parameters():
            param.requires_grad = False
        for param in self.policy_net.value_head.parameters():
            param.requires_grad = False
        
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.policy_net.parameters()),
            lr=self.pretrain_lr,
            weight_decay=0.01
        )

        print("Generating validation set...")
        val_exprs = [self._generate_random_expression() for _ in range(500)]  # 固定500个验证样本

        # 评估函数（核心）
        def evaluate_accuracy(exprs, step):
            self.policy_net.eval()  # 切换到评估模式
            correct_preds = {'start': 0, 'mid': 0, 'end': 0}
            total = 0
            errors = {'start': [], 'mid': [], 'end': []}
            
            with torch.no_grad():
                for expr_cards in exprs:
                    # 构建状态
                    self.env.selected_cards = expr_cards
                    self.env.symbolic_executor.reset()
                    for card in expr_cards:
                        self.env.symbolic_executor.execute_step(card, len(self.env.symbolic_executor.trace))
                    
                    state = self.env._get_state()
                    trace = self.env.symbolic_executor.get_full_trace()
                    if len(trace) < 3:
                        continue
                    
                    # 获取目标值
                    mid_point = len(trace) // 2
                    targets = {
                        'start': trace[0][1] / 100.0,
                        'mid': trace[mid_point][1] / 100.0,
                        'end': trace[-1][1] / 100.0 if trace[-1][1] is not None else 0.0
                    }
                    
                    # 模型预测
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    features = self.policy_net.get_features(state_t)
                    pred_start = self.policy_net.value_head_start(features).item()
                    pred_mid = self.policy_net.value_head_mid(features).item()
                    pred_end = self.policy_net.value_head(features).item()
                    
                    # 计算误差（相对误差 + 绝对误差）
                    for key in ['start', 'mid', 'end']:
                        true_val = targets[key]
                        pred_val = {'start': pred_start, 'mid': pred_mid, 'end': pred_end}[key]
                        
                        # 避免除零
                        rel_error = abs(pred_val - true_val) / (abs(true_val) + 1e-6)
                        abs_error = abs(pred_val - true_val)
                        
                        # 准确率标准：相对误差<5% 或 绝对误差<0.05
                        if rel_error < 0.05 or abs_error < 0.05:
                            correct_preds[key] += 1
                        
                        errors[key].append(rel_error)
                    
                    total += 1
            
            # 打印报告
            print(f"\n--- Validation at Step {step} ---")
            for key in ['start', 'mid', 'end']:
                acc = correct_preds[key] / max(total, 1) * 100
                avg_error = np.mean(errors[key]) * 100
                print(f"{key.upper():>5}: Acc={acc:5.1f}% | AvgRelErr={avg_error:5.1f}%")
            
            # 关键指标：END准确率（最难）
            end_acc = correct_preds['end'] / max(total, 1)
            self.policy_net.train()  # 恢复训练模式
            return end_acc
        
        pretrain_losses = []
        best_val_acc = 0.0
        patience_counter = 0
        
        # 批量生成表达式（减少环境交互）
        batch_exprs = []
        for _ in range(self.pretrain_steps):
            batch_exprs.append(self._generate_random_expression())
        
        pretrain_losses = []

        # 在训练前先做基准评估
        print("\n[Pre-training Baseline]")
        evaluate_accuracy(val_exprs, 0)
        
        for step, expr_cards in enumerate(batch_exprs):
            # 批量状态构建（避免多次_get_state）
            self.env.selected_cards = expr_cards
            self.env.symbolic_executor.reset()
            for card in expr_cards:
                self.env.symbolic_executor.execute_step(card, len(self.env.symbolic_executor.trace))
            
            state = self.env._get_state()
            
            # 前向传播（使用no_grad加速）
            with torch.no_grad():
                trace = self.env.symbolic_executor.trace
            
            if len(trace) < 3:
                continue

            # 获取轨迹监督信号
            trace = self.env.symbolic_executor.get_full_trace()
            if len(trace) < 3:
                continue

            # 选择3个关键点监督：起点、中点、终点
            mid_point = len(trace) // 2
            targets = {
                'start': trace[0][1] / 100.0,
                'mid': trace[mid_point][1] / 100.0,
                'end': trace[-1][1] / 100.0 if trace[-1][1] is not None else 0.0
            }
            
            # 前向传播
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            features = self.policy_net.get_features(state_t)  
            pred_value = self.policy_net.value_head(features)
            
            # 多尺度预测：让不同层预测不同尺度
            # features来自Transformer的不同深度
            pred_mid = self.policy_net.value_head_mid(features)
            pred_start = self.policy_net.value_head_start(features)
            
            # 复合损失
            loss = (F.mse_loss(pred_start, torch.tensor([targets['start']]).to(self.device)) * 0.2 +
                    F.mse_loss(pred_mid, torch.tensor([targets['mid']]).to(self.device)) * 0.3 +
                    F.mse_loss(pred_value, torch.tensor([targets['end']]).to(self.device)) * 0.5)
                
                # 梯度累积（稳定训练）
            if step % 4 == 0:
                optimizer.zero_grad()
            
            loss.backward()
            
            if step % 4 == 3 or step == self.pretrain_steps - 1:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.policy_net.parameters()), 
                    max_norm=0.5
                )
                optimizer.step()
                
                pretrain_losses.append(loss.item())
                
            if step % 100 == 0:
                start_loss = F.mse_loss(pred_start, torch.tensor([targets['start']]).to(self.device))
                mid_loss = F.mse_loss(pred_mid, torch.tensor([targets['mid']]).to(self.device))
                end_loss = F.mse_loss(pred_mid, torch.tensor([targets['end']]).to(self.device))
                avg_loss = np.mean(pretrain_losses[-100:])
                print(f"Start={start_loss:.4f} | Mid={mid_loss:.4f} | End={end_loss:.4f}")
                print(f"Pretrain Step {step}/{self.pretrain_steps}, Loss: {avg_loss:.4f}")

            # === 新增：定期评估 ===
            if step % 500 == 0 and step > 0:
                val_acc = evaluate_accuracy(val_exprs, step)
            
            
            if step % 100 == 0:
                start_loss = F.mse_loss(pred_start, torch.tensor([targets['start']]).to(self.device))
                mid_loss = F.mse_loss(pred_mid, torch.tensor([targets['mid']]).to(self.device))
                end_loss = F.mse_loss(pred_mid, torch.tensor([targets['end']]).to(self.device))
                avg_loss = np.mean(pretrain_losses[-100:])
                print(f"Step {step}: Start={start_loss:.4f} | Mid={mid_loss:.4f} | End={end_loss:.4f} | Avg={avg_loss:.4f}")
        
        # 最终评估
        print("\n[Final Validation]")
        evaluate_accuracy(val_exprs, step)
        
        # 解冻所有参数，准备PPO训练
        for param in self.policy_net.parameters():
            param.requires_grad = True
        
        print("✓ 符号预训练完成，损失:", np.mean(pretrain_losses[-100:]))
    
    def _generate_random_expression(self, min_len=3, max_len=9) -> List[Card]:
        """生成带括号的随机有效表达式"""
        while True:
            expr = []
            # 使用递归下降生成，确保括号平衡
            self._generate_recursive(expr, 0, min_len, max_len)
            
            if len(expr) >= min_len and self.env.is_valid_expression(expr):
                return expr
        
    def _generate_recursive(self, expr: List[Card], depth: int, min_len: int, max_len: int):
        """递归生成子表达式"""
        if len(expr) >= max_len:
            return
        
        # 随机选择生成：数字、二元运算或括号
        choice = random.random()
        
        if choice < 0.4 or len(expr) + 3 > max_len:
            # 生成数字
            expr.append(Card(CardType.NUMBER, value=random.randint(1, 10)))
        elif choice < 0.7 and depth < 2:
            # 生成括号表达式
            expr.append(Card(CardType.OPERATOR, operator_type=OperatorType.LEFTBRA))
            self._generate_recursive(expr, depth + 1, min_len, max_len)
            expr.append(Card(CardType.OPERATOR, operator_type=OperatorType.RIGHTBRA))
        else:
            # 生成二元运算
            if len(expr) == 0 or expr[-1].card_type != CardType.NUMBER:
                expr.append(Card(CardType.NUMBER, value=random.randint(1, 5)))
            
            op = random.choice([OperatorType.PLUS, OperatorType.MULTIPLY])
            expr.append(Card(CardType.OPERATOR, operator_type=op))
            
            # 递归生成右操作数
            self._generate_recursive(expr, depth, min_len, max_len)
    
    def train(self):
        """主训练循环"""
        print(f"Starting training on {self.device}")

        env_times = deque(maxlen=100)
        infer_times = deque(maxlen=100)
        train_times = deque(maxlen=100)

        # 阶段1：符号预训练
        if self.pretrain_steps > 0:
            self.pretrain_symbolic()
        
        while self.episode < self.config['total_episodes']:
            # 1. 收集rollout
            # 改用批量收集
            start = time.time()
            num_envs=self.config.get('num_envs', 8)
            episode_reward, episode_length = self.collect_rollout_batch(
                num_envs=num_envs
            )
            self.episode += num_envs  # 注意：一次收集多个episode
            env_time = time.time() - start
            env_times.append(env_time)

            # 2. 关键：在每个episode后检查并更新课程阶段
            self.update_curriculum_stage()
            
            # 2. 检查是否更新
            start = time.time()
            if self.collector.is_full() or self.episode >= self.config['total_episodes']:
                # 执行PPO更新
                loss = self.trainer.update(self.collector.get_batch())
                self.losses.append(loss)
                train_time = time.time() - start
                train_times.append(train_time)
            
                
                # 熵衰减
                self.entropy_coef = max(
                    self.entropy_min, 
                    self.entropy_coef * self.entropy_decay
                )
                self.trainer.entropy_coef = self.entropy_coef
                
                #记录buffer容量
                filled = self.collector.ptr

                # 清空collector
                self.collector.reset()
                
                # 日志
                print(f"Episode {self.episode}: "
                      f"Reward={episode_reward:.2f}, "
                      f"Loss={loss:.4f}, "
                      f"Entropy={self.entropy_coef:.4f}, "
                      f"Stage={self.current_stage}, "
                      f"Buffer={filled}/{self.collector.buffer_size}")
            
            # 3. 定期评估
            if self.episode % self.config.get('log_interval', 50) == 0:
                avg_reward = np.mean(self.episode_rewards)
                avg_length = np.mean(self.episode_lengths)
                avg_loss = np.mean(self.losses) if self.losses else 0
                print(f"\n=== Performance Stats ===")
                print(f"Env Avg: {np.mean(env_times):.3f}s | "
                    f"Infer: ~{np.mean(env_times)/self.config['num_envs']*1000:.1f}ms/env | "
                    f"Train: {np.mean(train_times):.3f}s")
                print(f"\n=== Episode {self.episode} ===")
                print(f"Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.1f} | Avg Loss: {avg_loss:.4f}\n")
                print(f"Expr Len: {np.mean(self.expr_stats['avg_expr_len']):.2f} | "
                f"Valid Rate: {np.mean(self.expr_stats['valid_exprs']):.2%} | "
                f"Skills: {np.mean(self.expr_stats['skill_triggers']):.2%} |"
                f"Bracket Balance: {np.mean(self.expr_stats['bracket_bal']):.3f} |"
                f"Bracket Sum:{np.sum(self.expr_stats['bracket_num'])} |"
                f"Zero Rate:{np.mean(self.expr_stats['zero_damage_raw']):.3f}")
            
            # 4. 保存检查点
            if self.episode % self.config.get('save_interval', 500) == 0:
                self.save_checkpoint(self.episode)
        
        self.save_checkpoint('final')
        print("Training completed!")
    
    def save_checkpoint(self, episode):
        """保存检查点"""
        torch.save({
            'episode': episode,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': list(self.episode_rewards),
            'losses': list(self.losses),
            'config': self.config
        }, f"ppo_{episode}.pth")
        print(f"✓ Checkpoint saved at episode {episode}")
    
    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = deque(checkpoint.get('episode_rewards', []), maxlen=100)
        self.losses = deque(checkpoint.get('losses', []), maxlen=100)
        self.episode = checkpoint.get('episode', 0)
        print(f"Loaded checkpoint from episode {self.episode}")

config = {
        'total_episodes': 10000,
        'max_episode_steps': 512,
        'lr': 5e-5,  # 标准PPO学习率
        'hidden_size': 512,
        'buffer_size': 8192,  # 固定长度
        'ppo_epochs': 4,
        'mini_batch_size': 512,
        'clip_param': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.05,
        'entropy_decay': 0.996,
        'entropy_min': 0.005,
        'gamma': 0.99,
        'lam': 0.95,
        'max_grad_norm': 1.0,
        'log_interval': 50,
        'save_interval': 1000,
        'adam_eps': 1e-5,
        'reward_scaling': 20.0,
        'use_transformer': True,  # 用于后期对比实验开关
        'd_model': 64,           # Transformer内部维度
        'nhead': 2,               # 注意力头数
        'transformer_layers': 1,  # 编码器层数
        'weight_decay': 0.01,
        'use_curriculum': True,     #课程学习
        'num_envs': 8,           # 并行环境数量（建议：CPU核心数-2）
                             # 例如：16核CPU -> 14个env
        }


if __name__ == "__main__":
    import time

    env = Game(config.get('use_curriculum', True))
    trainer = TrainingManager(env, config)
    trainer.train()
