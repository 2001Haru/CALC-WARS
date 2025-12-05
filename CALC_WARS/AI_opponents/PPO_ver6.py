import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, Any, Tuple
import random
from collections import deque
from .Game_env_PPO import Game, CardType, OperatorType

class PPOTransformer(nn.Module):
    """混合架构: MLP处理静态特征 + Transformer处理序列"""
    def __init__(self, obs_dim=121, action_dim=29, hidden_size=512, 
                 seq_len=9, d_model=128, nhead=4, num_layers=2):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model
        
        # 1. 静态特征编码器（输入79维）
        self.static_encoder = nn.Sequential(
            nn.Linear(79, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # 2. 序列特征编码器
        # 塔1: 卡牌索引嵌入 (0-13数字, 14-19运算符, 20=PAD)
        self.card_index_embedding = nn.Embedding(21, d_model, padding_idx=20)
        
        # 塔2: 卡牌类型嵌入 (0=数字,1=加减,2=乘除,3=括号,4=空)
        self.card_type_embedding = nn.Embedding(5, d_model, padding_idx=4)
        
        # [ADD] 可学习位置编码（替代固定编码）
        self.pos_encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        
        # [ADD] 序列有效性掩码（告诉Transformer哪些是填充位置）
        self.register_buffer('sequence_mask', 
                            torch.ones(1, seq_len, dtype=torch.bool))
        
        # Transformer编码器（轻量级）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=hidden_size,
            dropout=0.2,
            batch_first=True,  # 重要！输入格式为[batch, seq, feature]
            activation='gelu'    # 比ReLU更平滑
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # [ADD] 注意力池化层（替代mean池化，让模型自主学习重要性）
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
        
        # 3. 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + d_model, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
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
        """初始化策略同原代码"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

            elif isinstance(module, nn.Parameter):
                nn.init.normal_(module, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        输入: [batch, 121] 或 [121]
        输出: (logits [batch, 29], values [batch, 1])
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # 拆分特征
        static_features = x[:, :79]          # [batch, 79]
        sequence_compressed = x[:, 79:97]    # [batch, 18]
        
        # ---------------- 静态分支 ----------------
        static_encoded = self.static_encoder(static_features)  # [batch, hidden_size]
        
        # ---------------- 序列分支（双塔嵌入） ----------------
        # [MODIFY] 将18维向量reshape为[batch, 9, 2]
        # 维度说明: [卡牌索引(0-19), 卡牌类型(0-4)]
        sequence_tokens = sequence_compressed.reshape(-1, self.seq_len, 2)
        
        # [ADD] 安全裁剪，确保在词表范围内
        card_indices = sequence_tokens[:, :, 0].long().clamp(0, 20)  # 0-20
        card_types = sequence_tokens[:, :, 1].long().clamp(0, 4)     # 0-4
        
        # [ADD] 双塔分别嵌入
        index_emb = self.card_index_embedding(card_indices)  # [batch, 9, d_model]
        type_emb = self.card_type_embedding(card_types)      # [batch, 9, d_model]
        
        # [ADD] 语义融合（相加而非拼接，节省参数）
        seq_emb = index_emb + type_emb + self.pos_encoding   # [batch, 9, d_model]
        
        # [ADD] Transformer编码（添加key_padding_mask）
        # src_key_padding_mask: True表示需要被mask的位置
        # 当card_index=20(PAD)且type=4(空)时，该位置为填充
        padding_mask = (card_indices == 20) & (card_types == 4)  # [batch, 9]
        seq_encoded = self.transformer(
            seq_emb, 
            src_key_padding_mask=padding_mask
        )  # [batch, 9, d_model]
        
        # [ADD] 注意力池化（替代mean）
        # 计算每个位置的注意力权重
        attention_scores = self.attention_pool(seq_encoded)  # [batch, 9, 1]
        attention_weights = F.softmax(attention_scores.masked_fill(
            padding_mask.unsqueeze(-1), -1e4), dim=1)  # 屏蔽填充位置
        
        seq_pooled = torch.sum(seq_encoded * attention_weights, dim=1)  # [batch, d_model]
        
        # ---------------- 融合分支 ----------------
        combined = torch.cat([static_encoded, seq_pooled], dim=-1)
        features = self.fusion(combined)
        
        # ---------------- 输出头 ----------------
        logits = self.policy_head(features)
        values = self.value_head(features)
        
        # 安全裁剪
        logits = logits.masked_fill(logits < -1e4, -1e4)
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
                 ppo_epochs: int = 4,
                 mini_batch_size: int = 64,
                 clip_param: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 1.0,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 reward_scaling = 200.0):
        
        self.policy_net = policy_net
        self.optimizer = optimizer
        self.scaler = scaler  
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        self.reward_scaling = reward_scaling
    
    def compute_gae(self, rewards: np.ndarray, values: np.ndarray, 
                   dones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """高效向量化GAE计算"""
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)

        rewards = rewards / self.reward_scaling
        
        last_gae = 0
        last_value = values[-1] * (1 - dones[-1])
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * last_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae
            returns[t] = advantages[t] + values[t]
            
            last_gae = advantages[t]
            last_value = values[t]
        
        # 标准化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
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
            min_valid_logit = logits[mask_t].min().item()
            logits = logits.masked_fill(~mask_t, min_valid_logit - 50.0)  # 始终比最小合法值低50

            epsilon = 1e-6
            logits[mask_t] = logits[mask_t].clamp(-10, 10) + epsilon

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
            
            log_prob = dist.log_prob(action)
            
            return action.item(), log_prob.item(), values.squeeze().item()


class TrainingManager:
    """优化的训练管理器"""
    def __init__(self, env, config: Dict[str, Any]):
        self.env = env
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.cuda.amp.GradScaler()  # 混合精度缩放器
        
        # 网络
        self.policy_net = PPOTransformer(
            obs_dim=121,  
            action_dim=29,
            hidden_size=config.get('hidden_size', 512),
            seq_len=9,      # 必须与环境中的最大表达式长度一致
            d_model=config.get('d_model', 128),      
            nhead=config.get('nhead', 4),            
            num_layers=config.get('transformer_layers', 2) 
        ).to(self.device)

        # [ADD] 更新优化器（Transformer需要更低学习率）
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
            ppo_epochs=config.get('ppo_epochs', 4),
            mini_batch_size=config.get('mini_batch_size', 64),
            clip_param=config.get('clip_param', 0.2),
            value_coef=config.get('value_coef', 0.5),
            entropy_coef=config.get('entropy_coef', 0.01),
            gamma=config.get('gamma', 0.99),
            lam=config.get('lam', 0.95),
            max_grad_norm=config.get('max_grad_norm', 0.5)
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
    }
    
    def collect_rollout(self) -> Tuple[float, int]:
        """收集一个episode, 添加到collector"""
        state = self.env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done and episode_length < self.config.get('max_episode_steps', 200):
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

                if bracket_count > 0 and random.random() < 0.05:
                    print(f"[DEBUG] Expr: {[str(c) for c in self.env.selected_cards]}, "
                    f"Result: {result}, Valid: {result is not None}")

                if random.random() < 0.002:
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
            if self.collector.is_full() or self.episode >= self.config['total_episodes']:
                # 执行PPO更新
                loss = self.trainer.update(self.collector.get_batch())
                self.losses.append(loss)
                
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
                      f"Buffer={filled}/{self.collector.buffer_size}")
            
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
        'buffer_size': 4096,  # 固定长度
        'ppo_epochs': 4,
        'mini_batch_size': 256,
        'clip_param': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.05,
        'entropy_decay': 0.998,
        'entropy_min': 0.005,
        'gamma': 0.99,
        'lam': 0.95,
        'max_grad_norm': 1.0,
        'log_interval': 50,
        'save_interval': 1000,
        'adam_eps': 1e-5,
        'reward_scaling': 1.0,
        'use_transformer': True,  # 用于后期对比实验开关
        'd_model': 128,           # Transformer内部维度
        'nhead': 4,               # 注意力头数
        'transformer_layers': 2,  # 编码器层数
        'weight_decay': 0.01,
        }



if __name__ == "__main__":
    env = Game()
    trainer = TrainingManager(env, config)
    trainer.train()