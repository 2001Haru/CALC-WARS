import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
from Game_env_DQN import Game
import numpy as np
import random
from typing import List, Tuple, Dict, Any

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
        self.lstm = nn.LSTM(128, hidden_size, batch_first=True)
        
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
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # 特征提取
        features = self.feature_extractor(x)  # [batch, seq_len, 128]
        
        # LSTM处理
        lstm_out, hidden_out = self.lstm(features, hidden_state)  # [batch, seq_len, hidden_size]
        
        # 使用最后一个时间步
        last_out = lstm_out[:, -1, :]  # [batch, hidden_size]
        
        # 双头输出
        action_logits = self.policy_head(last_out)  # [batch, action_dim]
        state_values = self.value_head(last_out)   # [batch, 1]
        
        return action_logits, state_values.squeeze(-1), hidden_out
        

class ExperienceBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.episode_buffer = []
    
    def add_step(self, state, action, reward, log_prob, value, done, action_mask, player_id):
        """添加单步经验"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'log_prob': log_prob,
            'value': value,
            'done': done,
            'action_mask': action_mask,
            'player_id': player_id
        }
        self.episode_buffer.append(experience)
        
        if done or len(self.episode_buffer) >= 2000:
            if len(self.episode_buffer) > 0:
                self.buffer.append(self.episode_buffer.copy())
                print(f"回合已保存，步数: {len(self.episode_buffer)}")
            self.episode_buffer = []
    
    def sample_episodes(self, batch_size: int) -> List[List[Dict]]:
        """采样完整回合"""
        if len(self.buffer) < batch_size:
            return random.sample(list(self.buffer), len(self.buffer))
        return random.sample(list(self.buffer), batch_size)
    
    def __len__(self):
        return len(self.buffer)


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
                 max_grad_norm: float = 0.5):
        
        self.policy_net = policy_net
        self.optimizer = optimizer
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
    
    def compute_advantages(self, rewards: List[float], values: List[float], 
                          dones: List[bool], gamma: float = 0.99, lam: float = 0.95):
        """计算GAE优势函数"""
        advantages = []
        returns = []
        
        last_advantage = 0
        last_value = values[-1] * (1 - dones[-1])  # 终止状态价值为0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * last_value * (1 - dones[t]) - values[t]
            advantage = delta + gamma * lam * (1 - dones[t]) * last_advantage
            last_advantage = advantage
            last_value = values[t]
            advantages.insert(0, advantage)
            returns.insert(0, advantage + values[t])
        
        return advantages, returns
    
    def update(self, episodes_batch: List[List[Dict]]):
        """PPO更新步骤 - 处理双玩家数据"""
        """添加调试信息的PPO更新"""
        if not episodes_batch or len(episodes_batch) == 0:
            print("没有经验数据用于训练")
            return 0.0
        
        all_losses = []
        
        for epoch in range(self.ppo_epochs):
            # 随机打乱回合顺序
            random.shuffle(episodes_batch)
            
            for episode in episodes_batch:
                if len(episode) < 2:  # 需要至少2步
                    continue
                
                # 调试信息：打印回合数据统计
                print(f" 步数={len(episode)}, 奖励范围=[{min(s['reward'] for s in episode):.2f}, {max(s['reward'] for s in episode):.2f}]")

                # 按玩家分离数据
                player0_steps = [step for step in episode if step['player_id'] == 0]
                player1_steps = [step for step in episode if step['player_id'] == 1]
                
                # 分别处理每个玩家的轨迹
                for player_steps in [player0_steps, player1_steps]:
                    if len(player_steps) < 2:
                        continue
                    
                    # 按时间顺序排序
                    player_steps.sort(key=lambda x: x.get('step_idx', 0))
                    
                    # 提取数据
                    states = np.array([step['state'] for step in player_steps])
                    actions = np.array([step['action'] for step in player_steps])
                    old_log_probs = np.array([step['log_prob'] for step in player_steps])
                    rewards = np.array([step['reward'] for step in player_steps])
                    values = np.array([step['value'] for step in player_steps])
                    dones = np.array([step['done'] for step in player_steps])
                    action_masks = [step['action_mask'] for step in player_steps]

                    T = len(states)  # 序列长度
                    
                    # 计算优势
                    advantages, returns = self.compute_advantages(rewards, values, dones)
                    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
                    print(f'advan: {advantages},returns: {returns}')
                    
                    # 转换为张量
                    states_t = torch.FloatTensor(states).to(next(self.policy_net.parameters()).device)
                    actions_t = torch.LongTensor(actions).to(states_t.device)
                    old_log_probs_t = torch.FloatTensor(old_log_probs).to(states_t.device)
                    advantages_t = torch.FloatTensor(advantages).to(states_t.device)
                    returns_t = torch.FloatTensor(returns).to(states_t.device)
                    
                    
                    states_sequence = states_t.unsqueeze(0)  # [1, T, obs_dim] - 批次大小为1，序列长度为T

                    # 前向传播
                    logits, values_pred, _ = self.policy_net(states_sequence)
                    logits = logits.squeeze(0)  # [T, action_dim]
                    values_pred = values_pred.squeeze(0)  # [T]
                                    
                    # 应用动作掩码
                    for t in range(T):
                        mask_t = torch.BoolTensor(action_masks[t]).to('cuda')
                        
                        # 确保logits[t]和mask_t形状匹配
                        if logits[t].shape != mask_t.shape:
                            # 如果形状不匹配，调整logits[t]的形状
                            # 假设logits[t]的形状是[1, action_dim]或类似
                            if len(logits[t].shape) > 1:
                                # 如果是二维，需要压缩或调整掩码
                                mask_t = mask_t.unsqueeze(0)  # 将[29]变为[1, 29]
                            else:
                                # 如果logits[t]是一维但形状不匹配，可能是维度问题
                                mask_t = mask_t.view_as(logits[t])  # 强制形状匹配
                        
                        # 现在安全地应用掩码
                        logits[t] = logits[t].masked_fill(~mask_t, -1e9)
                    
                    # 计算新概率
                    dist = Categorical(logits=logits)
                    new_log_probs = dist.log_prob(actions_t)
                    entropy = dist.entropy().mean()
                    print(f'prob: {new_log_probs}')
                    
                    # 计算ratio
                    ratio = torch.exp(new_log_probs - old_log_probs_t)
                    print(f'ratio: {ratio}')
                    
                    # PPO损失
                    surr1 = ratio * advantages_t
                    surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages_t
                    policy_loss = -torch.min(surr1, surr2).mean()
                    print(f'policy_loss: {policy_loss}')
                    
                    # 价值损失
                    value_loss = F.mse_loss(values_pred, returns_t)
                    print(f'value_loss: {value_loss}')
                    
                    # 总损失
                    total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                    print(f'total: {total_loss}')
                    
                    # 反向传播
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    
                    all_losses.append(total_loss.item())
        
        return np.mean(all_losses) if all_losses else 0.0
    
    
class SelfPlayAgent:
    """自我对弈智能体 - 针对您的环境优化"""
    def __init__(self, policy_net: nn.Module, device: torch.device, seq_len: int = 10):
        self.policy_net = policy_net
        self.device = device
        self.seq_len = seq_len
        self.state_buffer = deque(maxlen=seq_len)  # 存储最近的状态序列
        self.hidden_state = None
    
    def reset_hidden_state(self, batch_size: int = 1):
        """重置LSTM隐藏状态和状态缓冲区"""
        self.hidden_state = (
            torch.zeros(self.policy_net.lstm_layers, batch_size, self.policy_net.hidden_size).to(self.device),
            torch.zeros(self.policy_net.lstm_layers, batch_size, self.policy_net.hidden_size).to(self.device)
        )
        self.state_buffer.clear()
    
    def _prepare_state_sequence(self, state: np.ndarray) -> torch.Tensor:
        """准备状态序列供LSTM使用"""
        self.state_buffer.append(state)
        
        # 如果缓冲区不满，用第一个状态填充
        '''while len(self.state_buffer) < self.seq_len:
            self.state_buffer.appendleft(state)'''
        
        # 转换为张量 [1, seq_len, state_dim]
        state_sequence = np.stack(list(self.state_buffer))
        return torch.FloatTensor(state_sequence).unsqueeze(0).to(self.device)
    
    def get_action(self, state: np.ndarray, action_mask: np.ndarray) -> Tuple[int, float, float]:
        """选择动作 - 使用状态序列"""
        state_t = self._prepare_state_sequence(state)  # [1, seq_len, state_dim]
        
        with torch.no_grad():
            logits, value, self.hidden_state = self.policy_net(state_t, self.hidden_state)
            
            # 应用动作掩码
            mask_t = torch.BoolTensor(action_mask).to(self.device)
            masked_logits = logits.masked_fill(~mask_t, -1e9)
            
            # 采样动作
            dist = Categorical(logits=masked_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            return action.item(), log_prob.item(), value.item()

class TrainingManager:
    """训练管理器 - 协调整个训练流程"""
    def __init__(self, env, config: Dict[str, Any]):
        self.env = env
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化组件
        self.policy_net = PPOWithLSTM(
            obs_dim= 90,
            action_dim= 29,
            hidden_size=config.get('hidden_size', 256),
            lstm_layers=config.get('lstm_layers', 1)
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=config.get('lr', 1e-4),
            eps=config.get('adam_eps', 1e-5)
        )
        
        self.buffer = ExperienceBuffer(capacity=config.get('buffer_size', 10000))
        self.trainer = PPOTrainer(
            policy_net=self.policy_net,
            optimizer=self.optimizer,
            ppo_epochs=config.get('ppo_epochs', 4),
            mini_batch_size=config.get('mini_batch_size', 64),
            clip_param=config.get('clip_param', 0.2),
            value_coef=config.get('value_coef', 0.5),
            entropy_coef=config.get('entropy_coef', 0.01)
        )
        
        self.agent = SelfPlayAgent(self.policy_net, self.device)
        
        # 训练记录
        self.episode_rewards = []
        self.losses = []
        self.episode = 0
    
    def collect_episode(self) -> float:
        """收集一个回合的经验"""
        state = self.env.reset()
        self.agent.reset_hidden_state()
        
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < self.config.get('max_steps', 1000):
            # 获取有效动作
            action_mask = self.env.get_valid_actions_mask()
            
            # 选择动作
            action, log_prob, value = self.agent.get_action(state, action_mask)
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            # 存储经验
            player_id = 0  # 根据你的游戏逻辑设置
            self.buffer.add_step(
                state=state, action=action, reward=reward,
                log_prob=log_prob, value=value, done=done,
                action_mask=action_mask, player_id=player_id
            )
            
            state = next_state
            total_reward += reward
            steps += 1
        
        self.episode_rewards.append(total_reward)
        return total_reward
    
    def train_step(self) -> float:
        """执行一次训练步骤"""
        if len(self.buffer) < self.config.get('min_buffer_size', 1):
            print(f"缓冲区不足，当前有{len(self.buffer)}个回合，需要{self.config.get('min_buffer_size', 1)}个")
            return 0.0
        
        # 采样经验
        episodes_batch = self.buffer.sample_episodes(
            batch_size=self.config.get('batch_size', 8)
        )
        
        # 训练
        loss = self.trainer.update(episodes_batch)
        self.losses.append(loss)
        
        return loss
    
    def run_training(self):
        """运行完整训练流程"""
        print(f"Starting training on {self.device}")
        
        for episode in range(self.config['total_episodes']):
            # 收集经验
            reward = self.collect_episode()
            print(f"Episode {episode}: 收集经验完成，奖励={reward:.2f}, 缓冲区大小={len(self.buffer)}")
        
            # 定期训练
            if episode % self.config.get('train_interval', 1) == 0:
                print(f"Episode {episode}: 尝试训练...")
                loss = self.train_step()
                print(f"Episode {episode}: 训练完成，loss={loss:.4f}")
            
            # 记录和保存
            if episode % self.config.get('log_interval', 10) == 0:
                avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                avg_loss = np.mean(self.losses[-100:]) if self.losses else 0
                print(f"Episode {episode}, Reward: {reward:.2f}, "
                      f"Avg Reward: {avg_reward:.2f}, Loss: {avg_loss:.4f}")
            
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

'''每个episodes新开一局, 每一步中获取环境90维状态向量与29维掩码, 神经网络策略头给出动作概率分布，我们采样概率选择最终动作
每一局设置最大操作数，超出操作数直接进入下一局。每进入一个新状态记录[state, mask, action, reward, old_probs, next_state, value, done, perspective]存入episodes_list, 一局结束episodes_list存入memory

self.agent.update()进行实际训练。训练过程是随机抽取memory中batchsize个数的episodes_list,
先用perspective分开双玩家记录, 然后stack拼接为时间序列[s_0,s_1,...], [a_0,a_1,...], [r_0,...], [old_p_0,...]...
所以每个episodes可以得到很多轨迹序列, 我们计算损失函数: L_t^{PPO}(θ) = L_t^{CLIP}(θ) - c1 * L_t^{VF}(θ) + c2 * S[π_θ](s_t),
where L_t^{CLIP}(θ) = E_t[ min(r_t(θ)*A_t, clip(r_t(θ), 1-ε, 1+ε)*A_t)]
            
首先使用GAE计算A_t。A_t^{GAE(γ, λ)} = Σ (γλ)^k * δ_{t+k}, where δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
具体操作是对于轨迹序列[s_0,s_1,s_2,...]计算[v_0,v_1,v_2,...], 再根据公式计算[δ_0,δ_1,...]
最后从后往前, 记最后的A_T为δ_T, 从t = T-1 到 0, A_t = δ_t + (γ * λ) * A_{t+1}

其次, 我们计算r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t), 进而得到L_t^{CLIP}(θ)
最后的，我们使用λ-回报。L_t^{VF}(θ) =  (V_θ(s_t) - V_t^{target})^2, 其中V_t^{target} = A_t + V(s_t)
而S[π_θ](s_t) = - Σ π_θ(a|s_t) * log(π_θ(a|s_t)), 这是为了鼓励探索设置的

把所有结果加起来得到loss_t, 再对一个episodes中所有loss_t求平均, 然后backward更新参数, 完成一次训练
            
'''

if __name__ == "__main__":
    # 配置参数
    config = {
        'total_episodes': 3000,
        'max_steps': 1000,
        'lr': 1e-4,
        'hidden_size': 256,
        'lstm_layers': 1,
        'buffer_size': 10000,
        'min_buffer_size': 1,
        'batch_size': 8,
        'ppo_epochs': 4,
        'mini_batch_size': 64,
        'clip_param': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'train_interval': 1,
        'log_interval': 10,
        'save_interval': 500
    }

    from Game_env_DQN import Game
    env = Game()
    
    # 创建训练管理器
    trainer = TrainingManager(env, config)
    
    # 开始训练
    trainer.run_training()