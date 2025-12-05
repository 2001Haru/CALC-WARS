import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, Any, Tuple
import random
from collections import deque
from Env_exp_3 import Game, CardType, OperatorType, Card
from typing import List, Dict, Tuple, Optional 


class PPOTransformer(nn.Module):
    """混合架构: MLP处理静态特征 + Transformer处理序列"""
    def __init__(self, obs_dim=128, action_dim=29, hidden_size=1024, 
                 seq_len=9, d_model=256, nhead=4, num_layers=3):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model
        
        # 1. 静态特征编码器（输入110维）
        self.static_encoder = nn.Sequential(
            nn.Linear(110, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
                
        # 2. 序列特征编码器
        self.card_index_embedding = nn.Embedding(21, d_model, padding_idx=20)
        self.card_type_embedding = nn.Embedding(5, d_model, padding_idx=4)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
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

        # 注意力池化层
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
        
        # 3. 特征融合层（输入维度修正）
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + d_model, hidden_size),  # 注意：这里只需要+d_model，因为seq_pooled已经是d_model
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 4. 策略和价值头
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
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

        for transformer in [self.card_transformer]:
            for layer in transformer.layers:
                # 使用正交初始化
                if hasattr(layer.self_attn, 'in_proj_weight'):
                    nn.init.orthogonal_(layer.self_attn.in_proj_weight, gain=1.0)
                    nn.init.constant_(layer.self_attn.in_proj_bias, 0)
                # 前馈网络
                if hasattr(layer, 'linear1'):
                    nn.init.orthogonal_(layer.linear1.weight, gain=0.5)
                if hasattr(layer, 'linear2'):
                    nn.init.orthogonal_(layer.linear2.weight, gain=0.5)


    def get_features(self, x: torch.Tensor):
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("WARNING: Input state contains NaN/Inf!")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        static_features = torch.cat([x[:, :86], x[:, 104:]], dim=-1)  # 110维
        static_encoded = self.static_encoder(static_features)         # MLP处理

        # 3. 序列分支：原始9×2序列
        seq_raw = x[:, 86:104].reshape(-1, 9, 2)  # [batch, 9, 2]
        
        # 为每个token创建独立嵌入
        # token维度0：卡牌索引(0-20) → embedding
        # token维度1：类型(0-4) → embedding
        card_indices = (seq_raw[:, :, 0] * 20).long().clamp(0, 20)
        card_types = (seq_raw[:, :, 1] * 4).long().clamp(0, 4)
        card_padding_mask = (card_indices == 20) | (seq_raw[:, :, 1] == 0.2)  # PAD标记

        # 如果全为padding，则强制第一个token为非padding
        all_padding = card_padding_mask.all(dim=1, keepdim=True)
        if all_padding.any():
            # 将第一个token标记为非padding
            card_padding_mask[all_padding.squeeze(), 0] = False

        index_emb = self.card_index_embedding(card_indices)  # [batch,9,128]
        type_emb = self.card_type_embedding(card_types)      # [batch,9,128]
        
        # 4. 融合序列表示
        card_seq_emb = index_emb + type_emb + self.pos_encoding
        
        # 4. Transformer编码
        card_encoded = self.card_transformer(
            card_seq_emb, 
            src_key_padding_mask=card_padding_mask
        )
        
        # 5. 注意力池化（修正：传递mask和mode）
        card_pooled = self._attention_pool(card_encoded, card_padding_mask, mode='card')

        # 6. 最终融合
        combined = torch.cat([static_encoded, card_pooled], dim=-1)
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
    """标准PPO收集器: 扁平化存储 + 固定长度"""
    def __init__(self, buffer_size: int = 2048):
        self.buffer_size = buffer_size
        self.reset()
    
    def reset(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.action_masks = []
        self.ptr = 0
    
    def add(self, state, action, log_prob, value, reward, done, mask):
        """添加单步经验（无轨迹结构）"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.action_masks.append(mask)
        self.ptr += 1

    def finalize(self, last_value: float = 0.0):
        """在episode结束时调用，为GAE提供V(s_{t+1})"""
        self.values.append(last_value)  # 现在values比rewards多1个，这是正确的！
    
    def is_full(self) -> bool:
        return self.ptr >= self.buffer_size
    
    def get_batch(self) -> Dict[str, np.ndarray]:
        """返回用于PPO训练的扁平化数组"""
        return {
            'states': np.array(self.states, dtype=np.float32),
            'actions': np.array(self.actions, dtype=np.int64),
            'log_probs': np.array(self.log_probs, dtype=np.float32),
            'values': np.array(self.values, dtype=np.float32),
            'rewards': np.array(self.rewards, dtype=np.float32),
            'dones': np.array(self.dones, dtype=np.float32),
            'masks': np.array(self.action_masks, dtype=bool),
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
    
    def get_action(self, state: np.ndarray, action_mask: np.ndarray, 
                   player_id:int = 0, deterministic: bool = False) -> Tuple[int, float, float]:
        """返回动作、log_prob、价值"""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mask_t = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, values = self.policy_net(state_t)
            
            # 应用掩码
            logits = logits.masked_fill(~mask_t, float('-inf'))
        
            # 防止数值不稳定
            logits = torch.clamp(logits, min=-1e5, max=1e3)

            # 对手玩家添加策略噪声
            if player_id == 1 and not deterministic:
                noise = torch.randn_like(logits) * 0.05  # 5%噪声
                noise = noise.masked_fill(~mask_t, 0)  # 屏蔽无效位置的噪声
                logits = logits + noise
            
            dist = Categorical(logits=logits)
            
            if deterministic:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()

            if not mask_t[0, action.item()]:
                # 如果采样到非法动作（极小概率），强制选择最高概率的合法动作
                valid_probs = dist.probs.masked_fill(~mask_t, 0)
                action = valid_probs.argmax(dim=-1)
            
            log_prob = dist.log_prob(action)
            
            return action.item(), log_prob.item(), values.squeeze().item()


class TrainingManager:
    """优化的训练管理器"""
    def __init__(self, env, config: Dict[str, Any]):
        self.env = env
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.cuda.amp.GradScaler()  # 混合精度缩放器
        self.Main_network_id = config.get('Main_net_id',0)
        
        # 将trainer引用注入环境（用于symbolic_execution_trace）
        self.env.trainer = self

        # 网络
        self.policy_net = PPOTransformer(
            obs_dim=128,  
            action_dim=29,
            hidden_size=config.get('hidden_size', 1024),
            seq_len=9,      # 必须与环境中的最大表达式长度一致
            d_model=config.get('d_model', 256),      
            nhead=config.get('nhead', 4),            
            num_layers=config.get('transformer_layers', 3) 
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
        buffer_size=config.get('buffer_size', 2048)
        self.main_collector = PPORolloutCollector(buffer_size=buffer_size)
        self.oppo_collector = PPORolloutCollector(buffer_size=buffer_size)
        
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
        'non_effect_rate':deque(maxlen=1000),   #完全无效果表达式率
    }


    
    def collect_rollout(self) -> Tuple[float, int]:
        """收集一个episode, 添加到主轨迹和对手轨迹collector"""
        state = self.env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        action_hist = {i:0 for i in range(29)}
        
        while not done and episode_length < self.config.get('max_episode_steps', 512):
            # 当前玩家
            current_pid = 0 if self.env.current_player == self.env.player1 else 1
            
            # 获取动作
            action_mask = self.env.get_valid_actions_mask()
            action, log_prob, value = self.agent.get_action(state, action_mask, current_pid)
            action_hist[action] += 1

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

                if bracket_count > 0 and random.random() < 0.0001:
                    print(f"[DEBUG] Expr: {[str(c) for c in self.env.selected_cards]}, "
                    f"Result: {result}, Valid: {result is not None}")

                if random.random() < 0.0001:
                    indices = []
                    for card in self.env.selected_cards:
                        if card.card_type == CardType.NUMBER:
                            indices.append(card.value)  # 数字直接使用值
                        elif card.card_type == CardType.OPERATOR:
                            indices.append(card.operator_type.value)  # 运算符使用枚举值

                    reachable_hint = self.env.state_vector[72:86]
                    hint = [-1 for i in range(14)]
                    for i in range(14):
                        if i % 2 == 0 and reachable_hint[i] != -1:
                            hint[i] = 17.0 * reachable_hint[i]
                        elif i % 2 == 1 and reachable_hint[i] != -1:
                            hint[i] = 17.0 * reachable_hint[i]

                    print(f'Reachable hint :{hint}')
                    print(f'Selected cards :{indices}')
                    print(f'Target :{[self.env.target.red_zone] + self.env.target.yellow_zone + self.env.target.blue_zone}')


                if random.random() < 0.0001:
                    print(f"[DEBUG] Expr: {[str(c) for c in self.env.selected_cards]}, "
                    f"Result: {result}, Valid: {result is not None}")
                
                # 执行动作
                next_state, reward, done, info = self.env.step(action)
                
                # 记录统计（使用step()之前的数据）
                self.expr_stats['avg_expr_len'].append(expr_length)
                self.expr_stats['valid_exprs'].append(float(is_valid))
                self.expr_stats['bracket_bal'].append(bracket_balance)
                self.expr_stats['bracket_num'].append(bracket_count)
                if is_valid:
                    self.expr_stats['zero_damage_raw'].append(1.0 if raw_damage == 0 else 0.0)
                    skill_triggered = 1.0 if info.get('skill_triggered') else 0.0
                    self.expr_stats['skill_triggers'].append(skill_triggered)
                    self.expr_stats['non_effect_rate'].append(1.0 if raw_damage == 0 and skill_triggered == 0.0 else 0.0)

            else:
                next_state, reward, done, _ = self.env.step(action)
            
            # 存储（对齐PPO论文的标志位：reward放在当前状态后）
            if current_pid == self.Main_network_id:
                collector = self.main_collector
            else:
                collector = self.oppo_collector

            collector.add(
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

        if random.random() <= 0.1:
            action_dist_str = ' | '.join([f"{i}:{c}" for i,c in action_hist.items() if c>0])
            print(f"Action distribution: {action_dist_str}")
        
        # 在episode结束时调用finalize
        if done:
            # 正常结束，V(s_final) = 0
            if self.main_collector.ptr > 0:
                self.main_collector.finalize(last_value=0.0)
            if self.oppo_collector.ptr > 0:
                self.oppo_collector.finalize(last_value=0.0)
        else:
            # 超时截断，需要计算V(s_{t+1})
            _, _, last_value_main = self.agent.get_action(state, self.env.get_valid_actions_mask(), player_id=self.Main_network_id)
            _, _, last_value_oppo = self.agent.get_action(state, self.env.get_valid_actions_mask(), player_id=1-self.Main_network_id)
            
            if self.main_collector.ptr > 0:
                self.main_collector.finalize(last_value=last_value_main)
            if self.oppo_collector.ptr > 0:
                self.oppo_collector.finalize(last_value=last_value_oppo)

        return episode_reward, episode_length
    
    def train(self):
        """主训练循环"""
        print(f"Starting training on {self.device}")

        while self.episode < self.config['total_episodes']:
            # 1. 收集rollout
            episode_reward, episode_length = self.collect_rollout()
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode += 1

            # 2. 检查是否更新
            if self.main_collector.is_full() or self.oppo_collector.is_full()\
                or self.episode >= self.config['total_episodes']:
                total_loss = 0
                update_count = 0

                # 执行PPO更新
                if self.main_collector.ptr > 0:
                    loss_main = self.trainer.update(self.main_collector.get_batch())
                    total_loss += loss_main
                    update_count += 1
                    
                if self.oppo_collector.ptr > 0:
                    loss_oppo = self.trainer.update(self.oppo_collector.get_batch())
                    total_loss += loss_oppo
                    update_count += 1

                # 平均loss
                avg_loss = total_loss / update_count if update_count > 0 else 0
                self.losses.append(avg_loss)
                    
                # 熵衰减
                self.entropy_coef = max(
                    self.entropy_min, 
                    self.entropy_coef * self.entropy_decay
                )
                self.trainer.entropy_coef = self.entropy_coef
                
                #记录buffer容量，主收集器和对手收集器应该是一样的
                filled = self.main_collector.ptr

                # 清空collector
                self.main_collector.reset()
                self.oppo_collector.reset()
                
                # 日志
                print(f"Episode {self.episode}: "
                      f"Reward={episode_reward:.2f}, "
                      f"Loss={avg_loss:.4f}, "
                      f"Entropy={self.entropy_coef:.4f}, "
                      f"Buffer={filled}/{self.main_collector.buffer_size}")
            
            # 3. 定期评估
            if self.episode % self.config.get('log_interval', 50) == 0:
                avg_reward = np.mean(self.episode_rewards)
                avg_length = np.mean(self.episode_lengths)
                avg_loss = np.mean(self.losses) if self.losses else 0
                print(f"\n=== Episode {self.episode} ===")
                print(f"Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.1f} | Avg Loss: {avg_loss:.4f}\n")
                print(f"Expr Len: {np.mean(self.expr_stats['avg_expr_len']):.2f} | "
                f"Valid Rate: {np.mean(self.expr_stats['valid_exprs']):.2%} | "
                f"Skills: {np.mean(self.expr_stats['skill_triggers']):.2%} |"
                f"Bracket Balance: {np.mean(self.expr_stats['bracket_bal']):.3f} |"
                f"Bracket Sum:{np.sum(self.expr_stats['bracket_num'])} |"
                f"Zero Rate:{np.mean(self.expr_stats['zero_damage_raw']):.3f} |"
                f"Useless Rate:{np.mean(self.expr_stats['non_effect_rate']):.3f} |")
            
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
        }, f"ppo_{episode}_2.pth")
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
        'lr': 3e-5,  # PPO学习率
        'hidden_size': 1024,
        'buffer_size': 2048,  # 固定长度
        'ppo_epochs': 4,
        'mini_batch_size': 128,
        'clip_param': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.05,
        'entropy_decay': 0.9975,
        'entropy_min': 0.001,
        'gamma': 0.996,
        'lam': 0.97,
        'max_grad_norm': 1.0,
        'log_interval': 50,
        'save_interval': 1000,
        'adam_eps': 1e-5,
        'reward_scaling': 100.0,
        'use_transformer': True,  # 用于后期对比实验开关
        'd_model': 256,           # Transformer内部维度
        'nhead': 4,               # 注意力头数
        'transformer_layers': 3,  # 编码器层数
        'weight_decay': 0.01,
        'use_curriculum': False,     #课程学习
        'Main_net_id': 0,           #当前训练网络id，由于这个版本是朴素自我对弈所以随便选一个
        }


if __name__ == "__main__":
    import time

    env = Game()
    trainer = TrainingManager(env, config)
    trainer.train()
