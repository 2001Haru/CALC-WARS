import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, Any, Tuple
import random
from collections import deque
from Game_env_PPO_old import Game

class PPO(nn.Module):
    """纯PPO网络:无序列维度"""
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        
        # 共享特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # 价值头
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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        输入: [batch, obs_dim] 或 [obs_dim]
        输出: (logits [batch, action_dim], values [batch, 1])
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        features = self.feature_extractor(x)
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
                 ppo_epochs: int = 4,
                 mini_batch_size: int = 64,
                 clip_param: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 1.0,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 reward_scaling = 20.0):
        
        self.policy_net = policy_net
        self.optimizer = optimizer
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
                
                # 前向传播
                logits, values = self.policy_net(batch_states)
                
                # 应用动作掩码
                logits = logits.masked_fill(~batch_masks, -1e9)
                
                # 策略损失
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失（使用clipping）
                values_pred_clipped = batch_old_values + torch.clamp(
                    values.squeeze(-1) - batch_old_values, -self.clip_param, self.clip_param
                )
                value_loss1 = F.mse_loss(values.squeeze(-1), batch_returns, reduction='none')
                value_loss2 = F.mse_loss(values_pred_clipped, batch_returns, reduction='none')
                value_loss = torch.max(value_loss1, value_loss2).mean()
                
                # 总损失
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
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
        mask_t = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, values = self.policy_net(state_t)
            
            # 应用掩码
            logits = logits.masked_fill(~mask_t, -1e10)

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
        
        # 网络
        self.policy_net = PPO(
            obs_dim=121,
            action_dim=29,
            hidden_size=config.get('hidden_size', 512)
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config.get('lr', 3e-4),
            eps=config.get('adam_eps', 1e-5)
        )
        
        # 训练器
        self.trainer = PPOTrainer(
            policy_net=self.policy_net,
            optimizer=self.optimizer,
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
                raw_damage = self.env.target.get_damage(result)
                is_valid = result is not None and result >= 0

                if random.random() < 0.001:
                    print(f"[DEBUG] Expr: {[str(c) for c in self.env.selected_cards]}, "
                    f"Result: {result}, Valid: {result is not None}")
                
                # 执行动作
                next_state, reward, done, info = self.env.step(action)
                
                # 记录统计（使用step()之前的数据）
                self.expr_stats['avg_expr_len'].append(expr_length)
                self.expr_stats['valid_exprs'].append(float(is_valid))
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
                
                # 清空collector
                self.collector.reset()
                
                # 日志
                print(f"Episode {self.episode}: "
                      f"Reward={episode_reward:.2f}, "
                      f"Loss={loss:.4f}, "
                      f"Entropy={self.entropy_coef:.4f}, "
                      f"Buffer={self.collector.ptr}/{self.collector.buffer_size}")
            
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
        }, f"ppo_checkpoint_{episode}.pth")
        print(f"✓ Checkpoint saved at episode {episode}")
    
    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = deque(checkpoint.get('episode_rewards', []), maxlen=100)
        self.losses = deque(checkpoint.get('losses', []), maxlen=100)
        self.episode = checkpoint.get('episode', 0)
        print(f"✓ Loaded checkpoint from episode {self.episode}")


# 使用示例
if __name__ == "__main__":
    config = {
        'total_episodes': 10000,
        'max_episode_steps': 512,
        'lr': 1e-4,  # 标准PPO学习率
        'hidden_size': 512,
        'buffer_size': 2048,  # 固定长度
        'ppo_epochs': 4,
        'mini_batch_size': 256,
        'clip_param': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.03,
        'entropy_decay': 0.9985,
        'entropy_min': 0.001,
        'gamma': 0.99,
        'lam': 0.95,
        'max_grad_norm': 1.0,
        'log_interval': 50,
        'save_interval': 1000,
        'adam_eps': 1e-5,
        'reward_scaling': 5.0
    }
    
    
    env = Game()
    trainer = TrainingManager(env, config)
    trainer.train()