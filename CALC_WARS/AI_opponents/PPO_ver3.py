import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
'''AI_opponents.'''
from .Game_env_DQN import Game
import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional

class PPOWithLSTM(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=256, lstm_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers

        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # LSTM记忆层
        self.lstm = nn.LSTM(128, hidden_size, lstm_layers, batch_first=True)
        
        # 策略和价值头
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

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor, hidden_state: Tuple[torch.Tensor, torch.Tensor] = None):
        # x shape: [batch, seq_len, obs_dim] 或 [batch, obs_dim]
        if len(x.shape) == 2:
            # 如果是单步输入，增加seq_len维度
            x = x.unsqueeze(1)  # [batch, 1, obs_dim]
        
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # 特征提取
        features = self.feature_extractor(x)  # [batch, seq_len, 128]
        
        # LSTM处理
        lstm_out, hidden_out = self.lstm(features, hidden_state)  # [batch, seq_len, hidden_size]
        
        # 使用所有时间步
        action_logits = self.policy_head(lstm_out)  # [batch, action_dim]
        
        # 价值头用所有时间步（每个状态都需要价值估计）
        state_values = self.value_head(lstm_out)  # [batch, seq_len, 1]
        
        # 返回序列价值和最后一个时间步的logits
        return action_logits, state_values.squeeze(-1), hidden_out  # [batch, seq_len]
    

class TrajectoryBuffer:
    def __init__(self, capacity: int = 100):
        self.capacity_per_player = capacity // 2  # 每个玩家分配一半容量
        self.buffer_p1 = deque(maxlen=self.capacity_per_player)
        self.buffer_p2 = deque(maxlen=self.capacity_per_player)
        self.current_traj_p1 = []
        self.current_traj_p2 = []
    
    def add_step(self, state, action, reward, log_prob, value, done,
                 action_mask, player_id, hidden_state):
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'log_prob': log_prob,
            'value': value,
            'done': done,
            'action_mask': action_mask,
            'player_id': player_id,
            'hidden_h': hidden_state[0].cpu().numpy() if hidden_state is not None else None,
            'hidden_c': hidden_state[1].cpu().numpy() if hidden_state is not None else None
        }
        
        # 根据玩家ID存储到不同轨迹
        if player_id == 0:
            self.current_traj_p1.append(experience)
            if done or len(self.current_traj_p1) >= 300:
                if self.current_traj_p1:
                    self.buffer_p1.append(self.current_traj_p1.copy())
                self.current_traj_p1 = []
        else:
            self.current_traj_p2.append(experience)
            if done or len(self.current_traj_p2) >= 300:
                if self.current_traj_p2:
                    self.buffer_p2.append(self.current_traj_p2.copy())
                self.current_traj_p2 = []
    
    def sample_trajectories(self, batch_size: int) -> List[List[Dict]]:
        """平衡采样两个玩家"""
        size_p1 = min(batch_size // 2, len(self.buffer_p1))
        size_p2 = min(batch_size - size_p1, len(self.buffer_p2))
        
        samples = []
        if size_p1 > 0:
            samples += random.sample(list(self.buffer_p1), size_p1)
        if size_p2 > 0:
            samples += random.sample(list(self.buffer_p2), size_p2)
        
        return samples
    
    def __len__(self):
        return len(self.buffer_p1) + len(self.buffer_p2)


class PPOTrainer:
    """PPO训练器 - 核心训练逻辑"""
    def __init__(self, 
                 policy_net: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 ppo_epochs: int = 4,
                 mini_batch_size: int = 64,
                 clip_param: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 2.0,
                 gamma: float = 0.99,
                 lam: float = 0.95):
        
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
    
    def compute_advantages(self, rewards: np.ndarray, values: np.ndarray, 
                          dones: np.ndarray):
        """计算GAE优势函数 - 向量化实现"""
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        last_advantage = 0
        last_value = values[-1] * (1 - dones[-1])
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * last_value * (1 - dones[t]) - values[t]
            advantage = delta + self.gamma * self.lam * (1 - dones[t]) * last_advantage
            last_advantage = advantage
            last_value = values[t]
            advantages[t] = advantage
            returns[t] = advantage + values[t]
        
        return advantages, returns
    
    def update(self, trajectories_batch: List[List[Dict]], entropy_coef: float = None) -> float:
        if not trajectories_batch:
            return 0.0
        
        if entropy_coef is None:
            entropy_coef = self.entropy_coef  # 保留实例属性作为后备
        
        all_losses = []
        device = next(self.policy_net.parameters()).device
        
        for _ in range(self.ppo_epochs):
            random.shuffle(trajectories_batch)
            
            for trajectory in trajectories_batch:
                if len(trajectory) < 2:
                    continue
                
                # ========== 数据准备 ==========
                states = np.array([step['state'] for step in trajectory])  # [seq_len, 90]
                actions = np.array([step['action'] for step in trajectory])  # [seq_len]
                old_log_probs = np.array([step['log_prob'] for step in trajectory])
                rewards = np.array([step['reward'] for step in trajectory])
                values_old = np.array([step['value'] for step in trajectory])
                dones = np.array([step['done'] for step in trajectory])
                masks = np.array([step['action_mask'] for step in trajectory])  # [seq_len, 29]
                
                # 计算GAE
                advantages, returns = self.compute_advantages(rewards, values_old, dones)
                advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
                
                # ========== Tensor转换 ==========
                seq_len = len(states)
                # states_seq: [1, seq_len, obs_dim]
                states_seq = torch.FloatTensor(states).unsqueeze(0).to(device)
                actions_seq = torch.LongTensor(actions).to(device)  # [seq_len]
                old_log_probs_seq = torch.FloatTensor(old_log_probs).to(device)  # [seq_len]
                advantages_seq = torch.FloatTensor(advantages).to(device)  # [seq_len]
                values_old_seq = torch.FloatTensor(values_old).to(device)
                returns_seq = torch.FloatTensor(returns).to(device)  # [seq_len]
                masks_seq = torch.BoolTensor(masks).to(device)  # [seq_len, 29]
                
                # ========== 隐藏状态恢复 ==========
                hidden = self._restore_hidden_state(trajectory[0], device)
                
                # ========== 前向传播 ==========
                # logits_last: [1, action_dim] - 策略只需要最后时刻
                # values_seq: [1, seq_len] - 价值需要所有时刻
                logits_seq, values_seq, _ = self.policy_net(states_seq, hidden)
                
                # 移除batch维度
                logits_seq = logits_seq.squeeze(0)  # [seq_len, action_dim]
                values_seq = values_seq.squeeze(0)  # [seq_len]
                
                # 2. 应用动作掩码（每个时间步独立）
                logits_seq = logits_seq.masked_fill(~masks_seq, -1e9)
                
                # 3. 价值序列squeeze batch维度
                values_seq = values_seq.squeeze(0)  # [seq_len]
                
                # ========== 策略损失 ==========
                dist = Categorical(logits=logits_seq)
                new_log_probs_seq = dist.log_prob(actions_seq)  # [seq_len]
                entropy = dist.entropy().mean()  # 平均熵
                
                ratio = torch.exp(new_log_probs_seq - old_log_probs_seq)  # [seq_len]
                surr1 = ratio * advantages_seq
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages_seq
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # ========== 价值损失 ==========
                # PPO的价值函数裁剪
                values_pred_clipped = values_old_seq + torch.clamp(
                    values_seq - values_old_seq, -self.clip_param, self.clip_param
                )
                value_losses = F.mse_loss(values_seq, returns_seq, reduction='none')
                value_losses_clipped = F.mse_loss(values_pred_clipped, returns_seq, reduction='none')
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
                
                # ========== 总损失 ==========
                total_loss = policy_loss + self.value_coef * value_loss - entropy_coef * entropy
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                all_losses.append(total_loss.item())

                if random.random() < 0.01:  # 1%概率打印，避免日志爆炸
                    lstm_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.policy_net.lstm.parameters(), 
                        float('inf')
                    )
                    print(f"LSTM梯度范数: {lstm_grad_norm:.4f}")
                    
                    # 检查P2路径
                    for name, param in self.policy_net.named_parameters():
                        if 'lstm' in name and param.grad is not None:
                            if param.grad.abs().mean() < 1e-6:
                                print(f"警告: {name} 梯度接近零！")

                        
        return np.mean(all_losses) if all_losses else 0.0

    def _restore_hidden_state(self, first_step: Dict, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """从轨迹第一步恢复隐藏状态"""
        hidden_h_np = first_step.get('hidden_h')
        hidden_c_np = first_step.get('hidden_c')
        
        if hidden_h_np is not None and hidden_c_np is not None:
            # hidden_h_np形状: [lstm_layers, 1, hidden_size]
            hidden_h = torch.FloatTensor(hidden_h_np).to(device)
            hidden_c = torch.FloatTensor(hidden_c_np).to(device)
            return (hidden_h, hidden_c)
        return None
    

class SelfPlayAgent:
    """自我对弈智能体"""
    def __init__(self, policy_net: nn.Module, device: torch.device):
        self.policy_net = policy_net
        self.device = device
        # 为每个玩家维护独立隐藏状态
        self.hidden_states = {
            0: None,  # player_id=0 (player1)
            1: None   # player_id=1 (player2)
        }
    
    def reset_hidden_state(self, batch_size: int = 1):
        """重置LSTM隐藏状态 - 确保P1和P2独立"""
        h1 = torch.zeros(self.policy_net.lstm_layers, batch_size, 
                        self.policy_net.hidden_size).to(self.device)
        c1 = torch.zeros_like(h1)
        h2 = torch.zeros(self.policy_net.lstm_layers, batch_size, 
                        self.policy_net.hidden_size).to(self.device)
        c2 = torch.zeros_like(h2)
        
        # 关键：使用不同的tensor对象
        self.hidden_states = {0: (h1, c1), 1: (h2, c2)}
    
    def get_action(self, state: np.ndarray, action_mask: np.ndarray, player_id: int) -> Tuple[int, float, float, Tuple]:
        """选择动作，返回隐藏状态用于训练"""
        state_t = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, obs_dim]
        
        with torch.no_grad():
            hidden_state = self.hidden_states[player_id]
            # 前向传播
            # logits: [1, action_dim]
            # values: [1, 1, seq_len=1] → 因为输入seq_len=1
            logits, values, new_hidden = self.policy_net(state_t, hidden_state)
            
            # ========== 提取序列中的唯一值 ==========
            # values形状是 [1, 1]，squeeze两次得到标量
            value = values.squeeze().item()  # 或者用 values[0,0] 更明确
            
            # 动作掩码处理不变
            mask_t = torch.BoolTensor(action_mask).to(self.device)
            masked_logits = logits.masked_fill(~mask_t, -1e9)
            
            # 采样
            dist = Categorical(logits=masked_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # 更新隐藏状态
            self.hidden_states[player_id] = new_hidden
            
            return action.item(), log_prob.item(), value, new_hidden
        
class TrainingManager:
    """训练管理器"""
    def __init__(self, env, config: Dict[str, Any]):
        self.env = env
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化网络
        self.policy_net = PPOWithLSTM(
            obs_dim=90,
            action_dim=30,
            hidden_size=config.get('hidden_size', 256),
            lstm_layers=config.get('lstm_layers', 2)
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=config.get('lr', 1e-3),
            eps=config.get('adam_eps', 1e-5)
        )
        
        self.buffer = TrajectoryBuffer(capacity=config.get('buffer_size', 100))
        self.trainer = PPOTrainer(
            policy_net=self.policy_net,
            optimizer=self.optimizer,
            ppo_epochs=config.get('ppo_epochs', 4),
            mini_batch_size=config.get('mini_batch_size', 64),
            clip_param=config.get('clip_param', 0.2),
            value_coef=config.get('value_coef', 0.5),
            entropy_coef=config.get('entropy_coef', 0.01),
            gamma=config.get('gamma', 0.99),
            lam=config.get('lam', 0.95)
        )
        
        self.agent = SelfPlayAgent(self.policy_net, self.device)
        
        # 训练记录
        self.episode_rewards = []
        self.losses = []
        self.episode = 0

        # 初始化熵系数
        self.initial_entropy_coef = config.get('entropy_coef', 0.01)
        self.entropy_coef = self.initial_entropy_coef  # 当前熵系数
        self.entropy_decay = config.get('entropy_decay', 0.99)  # 衰减率
        self.entropy_min = config.get('entropy_min', 0.001)  # 最小值保护
    
    def collect_episode(self) -> float:
        """收集一个回合的经验"""
        state = self.env.reset()
        self.agent.reset_hidden_state()
        
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < self.config.get('max_episode_steps', 300):
            # 获取当前玩家ID
            current_pid = 0 if self.env.current_player == self.env.player1 else 1
        
            action_mask = self.env.get_valid_actions_mask()
            
            # 传入player_id获取对应隐藏状态
            action, log_prob, value, hidden_state = self.agent.get_action(
                state, action_mask, current_pid
            )
            
            next_state, reward, done, info = self.env.step(action)
            
            # 存储经验：状态已经是当前玩家视角
            self.buffer.add_step(
                state=state, action=action, reward=reward,
                log_prob=log_prob, value=value, done=done,
                action_mask=action_mask, player_id=current_pid,
                hidden_state=hidden_state  # 该玩家的隐藏状态
            )
            
            state = next_state
            total_reward += reward
            steps += 1
        
        self.episode_rewards.append(total_reward)
        return total_reward
    
    def train_step(self) -> float:
        """执行一次训练步骤"""
        min_size = self.config.get('min_buffer_size')
        if len(self.buffer) < min_size:
            print(f"缓冲区不足，当前有{len(self.buffer)}条轨迹，需要{min_size}条")
            return 0.0
        
        # 采样经验
        batch_size = self.config.get('batch_size', 8)
        trajectories = self.buffer.sample_trajectories(batch_size)
        
        # 训练
        loss = self.trainer.update(trajectories, entropy_coef=self.entropy_coef)
        self.losses.append(loss)

        # 每次训练后衰减
        self.entropy_coef *= self.entropy_decay
        
        # 保护：不低于最小值
        self.entropy_coef = max(self.entropy_coef, self.entropy_min)
        
        # 每100次衰减打印一次（避免日志刷屏）
        if len(self.losses) % 100 == 0:
            print(f"熵系数已衰减至: {self.entropy_coef:.6f}")
        
        return loss
    
    def run_training(self):
        """运行完整训练流程"""
        print(f"Starting training on {self.device}")
        
        for episode in range(self.config['total_episodes']):
            # 收集经验
            reward = self.collect_episode()
            
            # 验证隐藏状态是否正确分离
            if episode % 100 == 0:
                print(f"[调试] Episode {episode}")
                print(f"  P1隐藏状态: {self.agent.hidden_states[0][0].abs().mean().item():.4f}")
                print(f"  P2隐藏状态: {self.agent.hidden_states[1][0].abs().mean().item():.4f}")

            # 定期训练
            if episode % self.config.get('train_interval') == 0 and len(self.buffer) >= self.config.get('min_buffer_size'):
                loss = self.train_step()
                print(f"Episode {episode}: 奖励={reward:.2f}, Loss={loss:.4f}, 缓冲区={len(self.buffer)}, 熵系数={self.entropy_coef:.6f}")
        
            # 记录和保存
            if episode % self.config.get('log_interval') == 0:
                avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                avg_loss = np.mean(self.losses[-100:]) if self.losses else 0
                print(f"=== Episode {episode} 统计 ===")
                print(f"Avg Reward (最近100): {avg_reward:.2f}")
                print(f"Avg Loss (最近100): {avg_loss:.4f}")
            
            if episode % self.config.get('save_interval', 500) == 0 and episode != 0:
                self.save_checkpoint(episode)
        
        self.save_checkpoint('final')
        print("Training completed!")
    
    def save_checkpoint(self, episode):
        """保存检查点"""
        checkpoint = {
            'episode': episode,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'losses': self.losses,
            'config': self.config
        }
        torch.save(checkpoint, f"ppo_checkpoint_{episode}.pth")
        print(f"Checkpoint saved at episode {episode}")
    
    def load_checkpoint(self, filepath):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.losses = checkpoint.get('losses', [])
        self.episode = checkpoint.get('episode', 0)
        print(f"Loaded checkpoint from episode {self.episode}")


if __name__ == "__main__":
    # 配置参数 - 优化后的值
    config = {
        'total_episodes': 5000,
        'max_episode_steps': 300,
        'lr': 1e-4,  # 略高的学习率
        'hidden_size': 256,
        'lstm_layers': 2,  # 使用2层LSTM
        'buffer_size': 20,
        'min_buffer_size': 8,  # 需要更多数据再开始训练
        'batch_size': 16,
        'ppo_epochs': 4,
        'mini_batch_size': 512,
        'clip_param': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'entropy_decay': 0.997,  # 每训练一次衰减
        'entropy_min': 0.001,
        'gamma': 0.99,
        'lam': 0.95,
        'train_interval': 2,  
        'log_interval': 50,
        'save_interval': 500,
        'adam_eps': 1e-5
    }

    from Game_env_DQN import Game
    env = Game()
    
    # 创建训练管理器
    trainer = TrainingManager(env, config)
    
    # 开始训练
    trainer.run_training()