import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, Any, Tuple
import random
import time
from collections import deque
from Env_sparse import Game, CardType, OperatorType, Card
from typing import List, Dict, Tuple, Optional 
import pickle
from pathlib import Path
import heapq
from Demonbot import DemonAgent
from Demonbot_2 import DemonAgent_V2
from Demonbot_3 import DemonAgent_V3
from Demonbot_4 import DemonAgent_V4
from Demonbot_5 import DemonAgent_V5
import os
os.environ["WANDB_API_KEY"] = "aa28905ac52e47e17f29015051d62a5c4b71d186"
import wandb

# --- 经验池与网络结构 ---
class GoldExperienceBuffer:
    """
    精英经验池：使用最小堆维护历史表现最好的 Top-K 个 Episode。
    只有总伤害超过堆顶元素的 Episode 才能进入。
    """
    def __init__(self, max_episodes: int = 200):
        self.max_episodes = max_episodes
        # 改用列表作为 FIFO 队列
        self.buffer = []  
        
        # 扁平化的缓存，用于快速采样
        self._flat_states = None
        self._flat_actions = None
        self._is_dirty = False

    def add_episode(self, episode):
        """尝试将 episode 加入精英池"""
        # 门槛1
        damage_ok = episode.total_damage > 0 #config.get('damage_threshold', 40)
        reward_ok = episode.episode_reward > 0 # config.get('reward_thresold', 500)

        if not (damage_ok or reward_ok):
            return
        
        # 压缩存储以节省内存
        ep_data = {
            'states': np.array(episode.states, dtype=np.float32),
            'actions': np.array(episode.actions, dtype=np.int64)
        }
        
        # FIFO 逻辑：直接添加，如果满了就移除最旧的
        self.buffer.append(ep_data)
        if len(self.buffer) > self.max_episodes:
            self.buffer.pop(0)
            
        self._is_dirty = True

    def _rebuild_cache(self):
        """重建扁平化缓存"""
        if not self.buffer:
            self._flat_states = np.empty((0, 92)) # 修正维度
            self._flat_actions = np.empty((0,))
            return

        all_states = []
        all_actions = []
        for data in self.buffer:
            all_states.append(data['states'])
            all_actions.append(data['actions'])
        
        # 一次性拼接，效率较高
        self._flat_states = np.concatenate(all_states, axis=0)
        self._flat_actions = np.concatenate(all_actions, axis=0)
        self._is_dirty = False
            
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """随机采样 batch"""
        if not self.buffer:
            return None, None

        # 只有在数据变脏时才重新构建扁平缓存
        # 在 collect_rollout 阶段 add_episode 可能调用多次，但 update 阶段 sample 调用更多
        # 这种 Lazy Rebuild 机制是高效的
        if self._is_dirty or self._flat_states is None:
            self._rebuild_cache()
            
        num_samples = len(self._flat_actions)
        if num_samples == 0:
            return None, None
            
        indices = np.random.choice(num_samples, batch_size, replace=True)
        
        batch_states = self._flat_states[indices]
        batch_actions = self._flat_actions[indices]
        
        return torch.FloatTensor(batch_states), torch.LongTensor(batch_actions)
    
    def __len__(self):
        return len(self.buffer)
    
    def get_min_score(self):
        return 0.0
    
    def get_avg_score(self):
        # FIFO 模式下不存储 score，返回 0 或估算值
        return 0.0

class OpponentPool:
    """管理历史策略的快照池"""
    def __init__(self, capacity: int = 10, pool_dir: str = "./opponent_pool"):
        self.capacity = capacity
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(exist_ok=True)
        self.min_age_before_elimination = config.get('min_age_before_elimination',40)
        
        self.opponents = []  # 按插入顺序，FIFO
        self._load_existing()
    
    def _load_existing(self):
        loaded_count = 0
        for pth in sorted(self.pool_dir.glob("opponent_*.pth")):
            metadata_path = pth.with_suffix('.meta')
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'rb') as f:
                        meta = pickle.load(f)
                    win_rate = meta.get('win_rate', 0.5)
                    uses = meta.get('uses', 0)
                    age = meta.get('age', 0)
                except Exception as e:
                    print(f"  Warning: Failed to load metadata from {metadata_path}: {e}")
                    win_rate, uses, age = 0.5, 0, 0
            else:
                win_rate, uses, age = 0.5, 0, 0
                meta = {'win_rate': win_rate, 'uses': uses, 'age': age}
                try:
                    with open(metadata_path, 'wb') as f:
                        pickle.dump(meta, f)
                except: pass
            
            self.opponents.append({
                'path': pth,
                'win_rate': win_rate,
                'uses': uses,
                'age': age
            })
            loaded_count += 1
        print(f"✓ Loaded {loaded_count} existing opponents")
    
    def _compute_diversity_score(self, new_policy_state_dict: dict) -> float:
        if not self.opponents: return 0.0
        total_distance = 0.0
        for opp in self.opponents:
            checkpoint = torch.load(opp['path'])
            opp_state_dict = checkpoint['policy_net_state_dict']
            distance = 0.0
            for key in new_policy_state_dict.keys():
                p1 = new_policy_state_dict[key].flatten()
                p2 = opp_state_dict[key].flatten()
                diff = (p1 - p2) / (torch.abs(p2) + 1e-8)
                distance += torch.sum(diff ** 2).item()
            total_distance += np.sqrt(distance) / len(p1)
        return total_distance / len(self.opponents)
    
    def add_opponent(self, policy_state_dict: dict, episode: int):
        if len(self.opponents) >= self.capacity:
            eliminable_opponents = [
                (idx, opp) for idx, opp in enumerate(self.opponents)
                if opp['age'] >= self.min_age_before_elimination
            ]
            if eliminable_opponents:
                idx_to_elim, oldest = min(eliminable_opponents, key=lambda x: x[1]['win_rate'])
                oldest['path'].unlink(missing_ok=True)
                oldest['path'].with_suffix('.meta').unlink(missing_ok=True)
                self.opponents.pop(idx_to_elim)
                print(f"  Removed opponent idx={idx_to_elim}")
            else:
                oldest = self.opponents.pop(0)
                oldest['path'].unlink(missing_ok=True)
                oldest['path'].with_suffix('.meta').unlink(missing_ok=True)
        
        save_path = self.pool_dir / f"opponent_{episode:06d}.pth"
        torch.save({'episode': episode, 'policy_net_state_dict': policy_state_dict}, save_path)
        
        self.opponents.append({
            'path': save_path,
            'win_rate': 0.5,
            'uses': 0,
            'age': 0
        })
        print(f"  Added opponent from episode {episode}")
    
    def sample_opponent(self, strategy: str = "oldest") -> Tuple[dict, int]:
        if not self.opponents: return None, -1
        for opp in self.opponents: opp['age'] += 1
        
        max_attempts = min(10, len(self.opponents))
        for _ in range(max_attempts):
            if strategy == "uniform": idx = random.randint(0, len(self) - 1)
            elif strategy == "weighted":
                weights = [opp['win_rate'] for opp in self.opponents]
                idx = random.choices(range(len(self)), weights=weights, k=1)[0]
            elif strategy == "oldest": idx = 0
            elif strategy == "latest": idx = len(self) - 1
            else: idx = 0
            
            opp_path = self.opponents[idx]['path']
            if opp_path.exists():
                try:
                    checkpoint = torch.load(opp_path, map_location='cpu')
                    self.opponents[idx]['uses'] += 1
                    return checkpoint['policy_net_state_dict'], idx
                except:
                    self.opponents.pop(idx)
                    continue
            else:
                self.opponents.pop(idx)
                continue
        return None, -1
    
    def update_stats(self, idx: int, won: bool):
        if 0 <= idx < len(self.opponents):
            opp = self.opponents[idx]
            opp['win_rate'] = opp['win_rate'] * 0.9 + (1.0 if won else 0.0) * 0.1
    
    def __len__(self): return len(self.opponents)

class ResBlock(nn.Module):
    """残差块：让网络可以堆得更深而不退化"""
    def __init__(self, size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(size, size),
            nn.LayerNorm(size),
            nn.ReLU(),
            nn.Dropout(0.1), # 防止过拟合
            nn.Linear(size, size),
            nn.LayerNorm(size)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.net(x)
        out += identity  # 核心：残差连接
        return self.relu(out)

"""
动作空间设计：
选牌数量 3张经济 3-5张标准
板块A 2种选牌数量 * 7个伤害区域 
板块B 2种选牌数量 * 20种特殊数字

阶乘 2 6 120
立方 8 27 64
平方 4 9 16 25 36 49 81 100 121 144 169 
24 0 1

板块C 6种技能+2种结束。
共14+40+8=62种动作。
"""

class CommanderNet(nn.Module):
    """分层神经网络大脑网络"""
    def __init__(self, obs_dim=92, action_dim=62, hidden_size=512, num_blocks=4):
        super().__init__()

        # 1. 初始特征映射 (把 92 维映射到高维空间)
        self.input_layer = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        # 2. 残差塔 (Reasoning Core)
        # 这里堆叠 num_blocks 个残差块，每个块包含 2 层 Linear
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_size) for _ in range(num_blocks)
        ])
        
        # 3. Heads (输出层)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 正交初始化对深层网络非常重要
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # 预处理 (防止 NaN)
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if len(x.shape) == 1: x = x.unsqueeze(0)
        
        # 1. Input Mapping
        x = self.input_layer(x)
        
        # 2. Residual Tower
        for block in self.res_blocks:
            x = block(x)
            
        # 3. Heads
        logits = self.policy_head(x)
        values = self.value_head(x)
        
        return logits, values


class Episode:
    """管理单局游戏中单个玩家的完整轨迹"""
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.masks: List[np.ndarray] = []
        
        self.is_finalized = False
        self.episode_reward = 0.0
        self.episode_length = 0
        self.total_damage = 0.0
        
    def add_step(self, state, action, log_prob, value, reward, done, mask):
        assert not self.is_finalized
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.masks.append(mask)
        self.episode_reward += reward
        self.episode_length += 1
    
    def finalize(self, last_value: float = 0.0):
        assert not self.is_finalized
        self.values.append(last_value)
        self.is_finalized = True
    
    def to_batch(self) -> Dict[str, np.ndarray]:
        """
        [关键修改] 将 values 切分为当前值和下一个状态的值，确保形状完全对齐
        """
        assert self.is_finalized
        v_np = np.array(self.values, dtype=np.float32)
        return {
            'states': np.array(self.states, dtype=np.float32),
            'actions': np.array(self.actions, dtype=np.int64),
            'log_probs': np.array(self.log_probs, dtype=np.float32),
            'values': v_np[:-1],      # 当前状态的值 V(s_t) -> 长度 N
            'next_values': v_np[1:],   # 下一个状态的值 V(s_t+1) -> 长度 N (含 bootstrapping)
            'rewards': np.array(self.rewards, dtype=np.float32),
            'dones': np.array(self.dones, dtype=np.bool_),
            'masks': np.array(self.masks, dtype=bool),
            'episode_reward': self.episode_reward,
            'episode_length': self.episode_length,
        }
    
    def copy(self):
        assert self.is_finalized
        copied = Episode(self.player_id)
        copied.states = self.states.copy()
        copied.actions = self.actions.copy()
        copied.log_probs = self.log_probs.copy()
        copied.values = self.values.copy()
        copied.rewards = self.rewards.copy()
        copied.dones = self.dones.copy()
        copied.masks = self.masks.copy()
        copied.is_finalized = True
        copied.episode_reward = self.episode_reward
        copied.episode_length = self.episode_length
        copied.total_damage = self.total_damage
        return copied
    
    def reset(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.masks = []
        self.is_finalized = False
        self.episode_reward = 0.0
        self.episode_length = 0
        self.total_damage = 0.0
    

class TrajectoryBuffer:
    """[修改] 不再成对存储，只存 Student Episode 用于 PPO"""
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.episodes: deque = deque(maxlen=capacity)
        
    def add_episode(self, episode: Episode):
        assert episode.is_finalized
        self.episodes.append(episode.copy())
    
    def sample_batch(self, recent_num: int = None) -> Optional[Dict[str, np.ndarray]]:
        if not self.episodes: return None
        
        all_data = {
            'states': [], 'actions': [], 'log_probs': [], 'values': [],
            'next_values': [], 'rewards': [], 'dones': [], 'masks': []
        }

        selection = list(self.episodes)[-recent_num:] if recent_num else self.episodes
        
        for ep in selection:
            batch = ep.to_batch()
            for key in all_data: 
                if key in batch:
                    all_data[key].append(batch[key])
        
        return {k: np.concatenate(v, axis=0) for k, v in all_data.items()}

    def is_full(self): return len(self.episodes) >= self.capacity
    def clear(self): self.episodes.clear()


class PPOTrainer:
    def __init__(self, policy_net: nn.Module, optimizer: torch.optim.Optimizer,
                 scaler: torch.cuda.amp.GradScaler, config: Dict[str, Any],
                 gold_buffer: GoldExperienceBuffer = None):
    
        self.config = config
        self.policy_net = policy_net
        self.optimizer = optimizer
        self.scaler = scaler  
        self.gold_buffer = gold_buffer
        
        self.mini_batch_size = config.get('mini_batch_size', 64)
        self.clip_param = config.get('clip_param', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.gamma = config.get('gamma', 0.99)
        self.lam = config.get('lam', 0.95)
        self.reward_scaling = config.get('reward_scaling', 50.0)
        self.bc_coef = config.get('bc_coef', 0.5) 
    
    def compute_gae(self, rewards, values, next_values, dones):
        """
        [关键修改] 使用 next_values 直接对齐，彻底解决形状不匹配和边界计算问题
        """
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        rewards = rewards / self.reward_scaling
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            # next_values[t] 已经是对应的 V(s_{t+1})
            # 如果 dones[t] 为 True，说明是回合结束，不进行 bootstrapping
            nv = next_values[t] * (1.0 - float(dones[t]))
            delta = rewards[t] + self.gamma * nv - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.lam * (1.0 - float(dones[t])) * last_gae
            returns[t] = advantages[t] + values[t]
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns
    
    
    def update(self, rollout: Dict[str, np.ndarray], ppo_epochs: int) -> Dict[str, float]:
        """
        更新策略并返回详细指标字典，修复了 EV 计算的形状问题
        """
        device = next(self.policy_net.parameters()).device
        
        # 1. 计算 GAE (现在传入对齐后的 values 和 next_values)
        advantages, returns_np = self.compute_gae(
            rollout['rewards'], rollout['values'], rollout['next_values'], rollout['dones']
        )
        
        # 2. 准备张量
        states = torch.FloatTensor(rollout['states']).to(device)
        actions = torch.LongTensor(rollout['actions']).to(device)
        old_log_probs = torch.FloatTensor(rollout['log_probs']).to(device)
        old_values = torch.FloatTensor(rollout['values']).to(device) # 已对齐 N
        adv_t = torch.FloatTensor(advantages).to(device)
        ret_t = torch.FloatTensor(returns_np).to(device)
        masks = torch.BoolTensor(rollout['masks']).to(device)
        
        # 3. 计算 Explained Variance (形状现在完美对齐 N=1527)
        ev = 0.0
        with torch.no_grad():
            y_true = returns_np
            y_pred = rollout['values']
            var_y = np.var(y_true)
            if var_y > 1e-8:
                ev = 1.0 - np.var(y_true - y_pred) / var_y
        
        metrics_acc = {
            'loss_total': [], 'loss_policy': [], 'loss_value': [], 
            'loss_entropy': [], 'loss_bc': []
        }
        
        # 4. PPO 迭代更新
        for _ in range(ppo_epochs):
            indices = torch.randperm(len(states))
            for start in range(0, len(states), self.mini_batch_size):
                end = min(start + self.mini_batch_size, len(states))
                if end - start < self.mini_batch_size // 2: continue

                idx = indices[start:end]
                self.optimizer.zero_grad()
                
                with torch.amp.autocast('cuda'):
                    logits, values = self.policy_net(states[idx])
                    logits = logits.masked_fill(~masks[idx], -1e4)
                    
                    dist = Categorical(logits=logits)
                    new_log_probs = dist.log_prob(actions[idx])
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_log_probs - old_log_probs[idx])
                    surr1 = ratio * adv_t[idx]
                    surr2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * adv_t[idx]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value Loss Clipped
                    v_pred = values.squeeze(-1)
                    v_old = old_values[idx]
                    v_clipped = v_old + torch.clamp(v_pred - v_old, -self.clip_param, self.clip_param)
                    value_loss = 0.5 * torch.max(F.mse_loss(v_pred, ret_t[idx]), F.mse_loss(v_clipped, ret_t[idx])).mean()
                    
                    # BC Loss
                    bc_loss = torch.tensor(0.0).to(device)
                    if self.gold_buffer is not None:
                        bc_s, bc_a = self.gold_buffer.sample(len(idx))
                        if bc_s is not None:
                            bc_logits, _ = self.policy_net(bc_s.to(device))
                            bc_loss = F.cross_entropy(bc_logits, bc_a.to(device))
                    
                    total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy + self.bc_coef * bc_loss

                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                metrics_acc['loss_total'].append(total_loss.item())
                metrics_acc['loss_policy'].append(policy_loss.item())
                metrics_acc['loss_value'].append(value_loss.item())
                metrics_acc['loss_entropy'].append(entropy.item())
                metrics_acc['loss_bc'].append(bc_loss.item())
        
        res = {k: np.mean(v) if v else 0.0 for k, v in metrics_acc.items()}
        res['ev'] = float(ev)
        return res


class SelfPlayAgent:
    def __init__(self, policy_net: nn.Module, device: torch.device,
                 opponent_pool: OpponentPool = None, trainer=None):
        self.device = device
        self.opponent_pool = opponent_pool
        self.trainer = trainer
        self.policy_net = policy_net  
        self.opponent_net = None
        self.current_opponent_idx = -1
        
        # 集成魔王机器人
        self.demon_v1 = DemonAgent()
        self.demon_v2 = DemonAgent_V2()
        self.demon_v3 = DemonAgent_V3()
        self.demon_v4 = DemonAgent_V4()
        self.demon_v5 = DemonAgent_V5()
        self.current_demon = None
        
        # 对手网络初始化
        if opponent_pool is not None :
            self.opponent_net = CommanderNet(
                obs_dim=92, action_dim=62,
                hidden_size= config.get('hidden_size',512),
                num_blocks= config.get('num_block',4)
            ).to(device)

    def load_opponent(self, opponent_state_dict: dict, idx: int):
        if self.opponent_net is not None:
            self.opponent_net.load_state_dict(opponent_state_dict)
            for param in self.opponent_net.parameters():
                param.requires_grad = False
            self.current_opponent_idx = idx
        else:
            print("Warning: opponent_net not initialized")
    
    def get_action(self, state: np.ndarray, action_mask: np.ndarray, 
                   player_id:int = 0, deterministic: bool = False) -> Tuple[int, float, float]:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mask_t = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            net = self.policy_net if player_id == 0 else self.opponent_net
            if net is None: net = self.policy_net
            
            logits, values = net(state_t)
            logits = logits.masked_fill(~mask_t, float('-inf'))
            logits = torch.clamp(logits, min=-1e5, max=1e3)
            dist = Categorical(logits=logits)
            
            if deterministic:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()

            if not mask_t[0, action.item()]:
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
        self.scaler = torch.amp.GradScaler('cuda')
        self.Main_network_id = config.get('Main_net_id',0)
        
        self.env.trainer = self

        self.policy_net = CommanderNet(
            obs_dim=92,  action_dim=62, hidden_size=config.get('hidden_size', 1024),
              num_blocks=config.get('num_block',4)
        ).to(self.device)

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=config.get('lr', 5e-5),
            eps=config.get('adam_eps', 1e-5),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        self.gold_buffer = GoldExperienceBuffer(max_episodes=config.get('gold_buffer',200))
        print("Initialized Elite Gold Experience Buffer (Min-Heap)")

        self.trainer = PPOTrainer(
            policy_net=self.policy_net, optimizer=self.optimizer,
            scaler=self.scaler, config = config, gold_buffer = self.gold_buffer
        )
        
        capacity =config.get('capacity', 20)
        self.TrajectoryBuffer = TrajectoryBuffer(capacity=capacity)
        
        self.episode_num = 0
        self.losses = deque(maxlen=100)
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

        self.episodes_0 = Episode(player_id = 0)
        self.episodes_1 = Episode(player_id = 1)
        
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.entropy_decay = config.get('entropy_decay', 0.995)
        self.entropy_min = config.get('entropy_min', 0.001)

        self.episodes_stats = {
            'solver_attempts': deque(maxlen=1000),   # 每局尝试调用 Solver 次数 (Action 0-53)
            'solver_success': deque(maxlen=1000),    # 每局Solver 成功找到解的次数
            'success_rate': deque(maxlen=1000),       # 每局平均成功率
            'expr_length': deque(maxlen=1000),       # 每局平均表达式长度
            'damage_dealt': deque(maxlen=1000),      # 每局总造成伤害平均
            'skill_triggered': deque(maxlen=1000),   # 每局平均技能触发次数 (Combo)
            'skill_used': deque(maxlen=1000),        # 每局平均主动技能使用次数
            'rounds': deque(maxlen=1000),             # 平均每局轮次数
            'turns': deque(maxlen=1000),               # 平均每局回合次数
            'action_hist': deque(maxlen=1000),          # 动作分布
            'counter_Demon_v1': deque(maxlen=1000),        # 与魔王v1交手
            'counter_Demon_v2': deque(maxlen=1000),        # 与魔王v2交手
            'counter_Demon_v3': deque(maxlen=1000),        # 与魔王v3交手
            'counter_Demon_v4': deque(maxlen=1000),        # 与魔王v4交手
            'counter_Demon_v5': deque(maxlen=1000),        # 与魔王v5交手
            'win_Demon_v1': deque(maxlen=1000) ,           # 打败魔王v1
            'win_Demon_v2': deque(maxlen=1000) ,           # 打败魔王v2
            'win_Demon_v3': deque(maxlen=1000) ,           # 打败魔王v3
            'win_Demon_v4': deque(maxlen=1000) ,           # 打败魔王v4
            'win_Demon_v5': deque(maxlen=1000) ,           # 打败魔王v5
        }

        
        self.use_opponent_pool = config.get('use_opponent_pool', True)
        self.opponent_pool = None
        self.opponent_sampling_strategy = config.get('opponent_strategy', 'oldest')
        self.opponent_update_interval = config.get('opponent_update_interval', 150)
        
        if self.use_opponent_pool:
            self.opponent_pool = OpponentPool(
                capacity=config.get('opponent_pool_size', 8),
                pool_dir=config.get('opponent_pool_dir', './opponent_pool')
            )
            print(f"Opponent pool directory: {self.opponent_pool.pool_dir.absolute()}")
            print(f"Found {len(self.opponent_pool)} opponents after initialization")
        
        self.agent = SelfPlayAgent(
            self.policy_net, self.device, self.opponent_pool, trainer=self,
        )
        self.opponent_match_results = deque(maxlen=100)

        # --- PHASE 2 LOGIC: Load Checkpoint if needed ---
        if self.config.get('run_phase') == 2 and self.config.get('checkpoint_path'):
            print(f"\n PHASE 2 INITIALIZATION: Loading checkpoint from {self.config['checkpoint_path']}")
            self.load_checkpoint(self.config['checkpoint_path'])
            # 重置 optimizer 状态以适应 Phase 2 的微调 (LR 已在 config 中更改)
            self.optimizer = optim.AdamW(
                self.policy_net.parameters(),
                lr=config.get('lr', 3e-5), # 使用 Phase 2 的低 LR
                eps=config.get('adam_eps', 1e-5),
                weight_decay=config.get('weight_decay', 0.01)
            )
            self.trainer.optimizer = self.optimizer
            print("✓ Optimizer reset for Phase 2 fine-tuning")

    def collect_rollout(self) -> Tuple[float, int]:
        """收集一个episode"""
        if self.use_opponent_pool and self.episode_num % 5 == 0: 
            opponent_weights, opp_idx = self.opponent_pool.sample_opponent(self.opponent_sampling_strategy)
            if opponent_weights is not None:
                self.agent.load_opponent(opponent_weights, opp_idx)

        # --- 魔王降临机制 ---
        is_scripted_opponent = False
        if random.random() < config.get('counter_demon_prob', 0.5): 
            is_scripted_opponent = True
            # 随机选择一代魔王
            # 观察到疑似的策略更新冲突，暂时只允许 v1 出场
            self.agent.current_demon = \
    random.choices([self.agent.demon_v1, self.agent.demon_v2, self.agent.demon_v3, self.agent.demon_v4, self.agent.demon_v5],
                                          weights = [0.0,0.0,0.0,0.0,1.0], k=1)[0]
                
            self.agent.current_demon.reset()
                
        state = self.env.reset()
        done = False
        action_hist = {i:0 for i in range(9)}
        player0_won = False

        while not done and (self.episodes_0.episode_length + self.episodes_1.episode_length) < self.config.get('max_episode_steps', 512):
            current_pid = 0 if self.env.current_player == self.env.player1 else 1
            model_mask = self.env.get_oracle_mask()
                
            # 双重保险：如果 Oracle 说全都不行（比如无牌可打），
            # 至少要允许 End the Round (61)
            if model_mask.sum() == 0:
                model_mask[61] = 1.0
            
            # 动作选择
            if current_pid == 1 and is_scripted_opponent:
                # 脚本对手：调用规则
                action = self.agent.current_demon.get_action(state)
                log_prob, value = 0.0, 0.0
                # 注意，魔王的prob和value都是0，因此绝对不能进入PPO池子学习！！
                
            else:   # 自然模型调用
                action, log_prob, value = self.agent.get_action(state, model_mask, current_pid)
                
            # 监控模型动作分布防止纳什均衡
            if current_pid == 0:
                if 0<= action <= 53:
                    action_hist[0] += 1 # 零是计算的意思
                else:
                    action_hist[action - 53] += 1

            # 环境接收动作改变状态
            next_state, reward, done, info = self.env.step(action)

            # 从 Info 中解析伤害并累加到 Episode 对象
            step_damage = 0
            if 'message' in info and info['message'].startswith("Damage"):
                try:
                    step_damage = int(float(info['message'].split()[-1]))
                except: pass
            
            episodes = self.episodes_0 if current_pid == 0 else self.episodes_1
            episodes.total_damage += step_damage
    
            episodes.add_step(
                state=state, action=action, log_prob=log_prob, value=value,
                reward=reward, done=done, mask=model_mask
            )
            state = next_state

        if self.env.stats['solver_attempts'] > 0:
            avg_success = self.env.stats['solver_success'] / self.env.stats['solver_attempts']
            avg_length = self.env.stats['expr_length'] / self.env.stats['solver_success']
            self.episodes_stats['success_rate'].append(avg_success)
            self.episodes_stats['expr_length'].append(avg_length) # 长度统计同理，没出牌就不应该记入长度
        else:
            # 如果这局没用过 Solver，就不记录到成功率统计队列中，以免拉低平均值
            pass
        
        self.episodes_stats['solver_attempts'].append(self.env.stats['solver_attempts'])   # 每局尝试调用 Solver 次数 (Action 0-53)
        self.episodes_stats['solver_success'].append(self.env.stats['solver_success'])    # 每局Solver 成功找到解的次数
        self.episodes_stats['damage_dealt'].append(self.env.stats['damage_dealt'])      # 每局总造成伤害平均
        self.episodes_stats['skill_triggered'].append(self.env.stats['skill_triggered'])   # 每局平均技能触发次数 (Combo)
        self.episodes_stats['skill_used'].append(self.env.stats['skill_used'])        # 每局平均主动技能使用次数
        self.episodes_stats['rounds'].append(self.env.stats['rounds'])                  # 每局轮次数
        self.episodes_stats['turns'].append(self.env.stats['turns'])                  # 每局回合数
        self.episodes_stats['action_hist'].append(action_hist)                      # 动作分布 
        
        if done:
            lose_penalty = -1.0
            if self.env.player1.hp <= 0 and current_pid != 0:
                # 给 P0 补一刀
                # 我们添加一个虚拟步，State 为当前的死亡状态，Reward 为惩罚
                # Action 可以填一个无意义占位符 (如 60 End Turn)
                # Mask 全 0
                self.episodes_0.add_step(
                    state=state, # 死亡时的遗照
                    action=61,   # 占位动作
                    log_prob=0.0,
                    value=0.0,
                    reward=lose_penalty, # <--- 关键！注入痛感
                    done=True,
                    mask=np.zeros(62)
                )

            # 3. 检查 Player 1 (对手) 是否被动死亡 (用于统计)
            # 条件：P1 死了，且最后行动的不是 P1
            if self.env.player2.hp <= 0 and current_pid != 1:
                self.episodes_1.add_step(
                    state=state,
                    action=61,
                    log_prob=0.0,
                    value=0.0,
                    reward=lose_penalty,
                    done=True,
                    mask=np.zeros(62)
                )
            if self.env.player1.hp > 0: player0_won = True
            if self.episodes_0.episode_length > 0: self.episodes_0.finalize(last_value= 0.0)
            if self.episodes_1.episode_length > 0: self.episodes_1.finalize(last_value= 0.0)
        else:
            player0_won = self.env.player1.hp > self.env.player2.hp
            model_mask = self.env.get_oracle_mask()
                
            # 双重保险：如果 Oracle 说全都不行（比如无牌可打），
            # 至少要允许 End the Round (61)
            if model_mask.sum() == 0:
                model_mask[61] = 1.0
            _, _, last_value_main = self.agent.get_action(state, model_mask, player_id=self.Main_network_id)
            _, _, last_value_oppo = self.agent.get_action(state, model_mask, player_id=1-self.Main_network_id)
            if self.episodes_0.episode_length > 0: self.episodes_0.finalize(last_value= last_value_main)
            if self.episodes_1.episode_length > 0: self.episodes_1.finalize(last_value= last_value_oppo)

        if is_scripted_opponent:
            ver = self.agent.current_demon.name.split('_')[-1]          # 'v1'/'v2'/'v3'/'v4'/'v5'
            for v in ('v1', 'v2', 'v3', 'v4','v5'):
                flag = 1.0 if v == ver else 0.0
                self.episodes_stats[f'counter_Demon_{v}'].append(flag)
                self.episodes_stats[f'win_Demon_{v}'].append(flag if player0_won else 0.0)
        else:
            for v in ('v1', 'v2', 'v3', 'v4','v5'):
                self.episodes_stats[f'counter_Demon_{v}'].append(0.0)
                self.episodes_stats[f'win_Demon_{v}'].append(0.0)
        
        
        # --- 黄金回放核心逻辑 ---
        # 1. 主模型 (Player 0) -> TrajectoryBuffer (PPO) + GoldBuffer (BC)

        # 规则 1: 只要当前模型 (Player 0) 赢了，无论对手是谁，轨迹入池 (Self-Imitation)
        # 特攻魔王
        if player0_won and self.episodes_0.episode_reward > 0 and is_scripted_opponent:
            self.gold_buffer.add_episode(self.episodes_0)

        if self.episodes_0.episode_length > 0:
            self.TrajectoryBuffer.add_episode(self.episodes_0)
        else:
            if self.env.player1.hp <= 0:
                # OTK对局，一般是魔王直接给主模型秒了，卡牌减少之后超级罕见，因为7张牌秒杀绝对不可能
                print(f"Episode {self.episode_num}: OTK detected (Bad Luck). Ignoring stats.")


        # 2. 对手模型 (Player 1): 智能判断
        if self.episodes_1.episode_length > 0:
            if is_scripted_opponent:
                # 魔王数据：由于现在魔王动作完全不同而且是静态策略，可以视为专家模型
                # 魔王只是作为一个预先写好的决策树对手存在
                # Case B: 魔王赢了模型 -> 魔王的教学局，不一定存
                if not player0_won and self.episodes_1.episode_reward > 0 and \
                    self.agent.current_demon == self.agent.demon_v1:
                    self.gold_buffer.add_episode(self.episodes_1)
            else:
                # 自然对手 (Neural Network) 数据
                # 由于PPO是On Policy策略，我们的历史版本对手不能给PPO学习，但是BC是Off Policy的
                '''if not player0_won and self.episodes_1.episode_reward > 0:
                    self.gold_buffer.add_episode(self.episodes_1)'''
                pass    # 暂时不学习防止拟合到过去策略

        if self.agent.current_opponent_idx >= 0:
            opponent_won = not player0_won
            self.opponent_pool.update_stats(self.agent.current_opponent_idx, opponent_won)
            self.opponent_match_results.append(1 if player0_won else 0)

    def train(self):
        print(f"Starting training on {self.device}")
        if self.use_opponent_pool and len(self.opponent_pool) == 0:
            print("No Opponent Found. Add Self...")
            self.opponent_pool.add_opponent(self.policy_net.state_dict(), self.episode_num)

        start_time = time.time()
        while self.episode_num < self.config['total_episodes']:
            self.collect_rollout()
            epoch_metrics = []
    
            # 仅当玩家0至少行动过一次时，才计入奖励和长度统计
            # 如果是OTK局 (episodes0.length=0)，直接忽略，不污染平均值，哪怕非常少
            if self.episodes_0.episode_length > 0:
                self.episode_rewards.append(self.episodes_0.episode_reward)
                self.episode_lengths.append(self.episodes_0.episode_length + self.episodes_1.episode_length)
            
            self.episodes_0.reset()
            self.episodes_1.reset()
            self.episode_num += 1

            total_loss = 0
            update_count = 0

            # 更新策略
            # 小更新: 只需要从 TrajectoryBuffer 采样 (里面都是 Student 数据)
            if self.episode_num % config.get('small_update_interval') == 0 and not self.TrajectoryBuffer.is_full():
                recent_num = config.get('small_update_recent')
                batch = self.TrajectoryBuffer.sample_batch(recent_num=recent_num)
                if batch is not None:
                    res = self.trainer.update(batch, ppo_epochs=config.get('ppo_epochs_lil'))
                    epoch_metrics.append(res)

            # 大更新: 采样全部 TrajectoryBuffer 并清空
            if self.TrajectoryBuffer.is_full() or self.episode_num >= self.config['total_episodes']:
                batch = self.TrajectoryBuffer.sample_batch()
                if batch is not None:
                    res = self.trainer.update(batch, ppo_epochs=config.get('ppo_epochs_tot'))
                    epoch_metrics.append(res)
                self.TrajectoryBuffer.clear()


            if epoch_metrics:
                # 聚合指标
                keys = epoch_metrics[0].keys()
                avg_m = {k: np.mean([m[k] for m in epoch_metrics]) for k in keys}
                
                self.losses.append(avg_m['loss_total'])

                # 上传 WandB
                wandb.log({
                    "Loss/Total": avg_m['loss_total'],
                    "Loss/Policy": avg_m['loss_policy'],
                    "Loss/Value": avg_m['loss_value'],
                    "Loss/BC": avg_m['loss_bc'],
                    "Loss/Entropy": avg_m['loss_entropy'],
                    "Value/Explained_Variance": avg_m['ev'], # 监控关键：解释方差
                    "Value/Entropy_Coef": self.entropy_coef,
                    "Training/BC_Coef": self.trainer.bc_coef,
                    "Training/Gold_Buffer_Size": len(self.gold_buffer),
                }, step=self.episode_num)
      
            self.entropy_coef = max(self.entropy_min, self.entropy_coef * self.entropy_decay)
            self.trainer.entropy_coef = self.entropy_coef
                
            if self.episode_num % config.get('print_check_interval',100) == 0:
                # 日志
                print(f"Episode {self.episode_num}: "
                    f"Reward={self.episode_rewards[-1]:.2f}, "
                    f"Loss={self.losses[-1]:.4f}, "
                    f"Entropy={self.entropy_coef:.4f}, "
                    )
                
            if self.episode_num % self.config.get('log_interval', 200) == 0:
                # 1. 计算基础指标
                avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
                win_rate = np.mean(self.opponent_match_results) if self.opponent_match_results else 0.5
                stats = self.episodes_stats
                has_attempts = bool(stats['solver_attempts'])

                rates = {}
                for v in ('v1', 'v2', 'v3', 'v4','v5'):
                    counter = np.sum(stats[f'counter_Demon_{v}']) if has_attempts else 0
                    win     = np.sum(stats[f'win_Demon_{v}'])     if has_attempts else 0
                    rates[v] = win / counter if counter else 0.0

                # 如果后面还要单独变量名，再解包即可
                win_Demon_rate_v1 = rates['v1']
                win_Demon_rate_v2 = rates['v2']
                win_Demon_rate_v3 = rates['v3']
                win_Demon_rate_v4 = rates['v4']
                win_Demon_rate_v5 = rates['v5']

                # 2. 计算 Commander 特有统计指标
                stat_success_rate = np.mean(self.episodes_stats['success_rate']) if self.episodes_stats['success_rate'] else 0
                stat_expr_len = np.mean(self.episodes_stats['expr_length']) if self.episodes_stats['expr_length'] else 0
                stat_damage = np.mean(self.episodes_stats['damage_dealt']) if self.episodes_stats['damage_dealt'] else 0
                stat_skill_trig = np.mean(self.episodes_stats['skill_triggered']) if self.episodes_stats['skill_triggered'] else 0
                stat_skill_used = np.mean(self.episodes_stats['skill_used']) if self.episodes_stats['skill_used'] else 0
                stat_rounds = np.mean(self.episodes_stats['rounds']) if self.episodes_stats['rounds'] else 0
                stat_turns = np.mean(self.episodes_stats['turns']) if self.episodes_stats['turns'] else 0
                stat_solver_atmpt = np.mean(self.episodes_stats['solver_attempts']) if self.episodes_stats['solver_attempts'] else 0
                

                # 3. WandB Log
                wandb.log({
                    "Performance/Avg_Reward": avg_reward,
                    "Performance/Avg_Length": avg_length,
                    "Performance/Win_Rate": win_rate,
                    "Performance/Win_Demon_Rate": win_Demon_rate_v1,
                    "Performance/Win_Demon_Rate_v2": win_Demon_rate_v2,
                    "Performance/Win_Demon_Rate_v3": win_Demon_rate_v3,
                    "Performance/Win_Demon_Rate_v4": win_Demon_rate_v4,
                    "Performance/Win_Demon_Rate_v5": win_Demon_rate_v5,
                    
                    "Commander/Solver_Success_Rate": stat_success_rate,
                    "Commander/Solver_Attempts": stat_solver_atmpt,
                    "Commander/Avg_Expr_Length": stat_expr_len,
                    "Commander/Avg_Damage": stat_damage,
                    "Commander/Skill_Combo_Count": stat_skill_trig,
                    "Commander/Active_Skill_Usage": stat_skill_used,
                    "Commander/Avg_Rounds": stat_rounds,
                    "Commander/Avg_Turns": stat_turns,
                }, step=self.episode_num)
            
            if self.episode_num % self.config.get('print_interval', 50) == 0:
                now_time = time.time()
                duration_hour = (now_time-start_time) / 3600
                
                # 重新获取最新均值用于打印
                avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
                avg_loss = np.mean(self.losses) if self.losses else 0
                
                # Stats needed for print
                p_success = np.mean(self.episodes_stats['success_rate']) if self.episodes_stats['success_rate'] else 0
                p_len = np.mean(self.episodes_stats['expr_length']) if self.episodes_stats['expr_length'] else 0
                p_damage = np.mean(self.episodes_stats['damage_dealt']) if self.episodes_stats['damage_dealt'] else 0
                p_skill_trig = np.mean(self.episodes_stats['skill_triggered']) if self.episodes_stats['skill_triggered'] else 0
                p_skill_used = np.mean(self.episodes_stats['skill_used']) if self.episodes_stats['skill_used'] else 0
                p_rounds = np.mean(self.episodes_stats['rounds']) if self.episodes_stats['rounds'] else 0
                p_turns = np.mean(self.episodes_stats['turns']) if self.episodes_stats['turns'] else 0

                # 类似于 WandB 的打印
                stats = self.episodes_stats
                has_attempts = bool(stats['solver_attempts'])

                rates = {}
                counter = {}
                for v in ('v1', 'v2', 'v3', 'v4','v5'):
                    counter[v] = np.sum(stats[f'counter_Demon_{v}']) if has_attempts else 0
                    win     = np.sum(stats[f'win_Demon_{v}'])     if has_attempts else 0
                    rates[v] = win / counter[v] if counter[v] else 0.0

                # 如果后面还要单独变量名，再解包即可
                p_win_Demon_rate_v1 ,p_counter_Demon_v1 = rates['v1'], counter['v1']
                p_win_Demon_rate_v2 ,p_counter_Demon_v2 = rates['v2'], counter['v2']
                p_win_Demon_rate_v3 ,p_counter_Demon_v3 = rates['v3'], counter['v3']
                p_win_Demon_rate_v4 ,p_counter_Demon_v4 = rates['v4'], counter['v4']
                p_win_Demon_rate_v5 ,p_counter_Demon_v5 = rates['v5'], counter['v5']

                # 修复: 动作历史的正确处理
                values = np.array([list(d.values()) for d in self.episodes_stats['action_hist']], dtype=np.float32)
                mean_ = values.mean(axis=0) # shape=(9,)
                # 将 numpy float 转为标准 float 以便格式化
                mean_dict = {k: float(v) for k, v in enumerate(mean_)}

                print(f"\n=== Episode {self.episode_num} === | Duration hour:{duration_hour:.2f}")
                print(f"Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.1f} | Avg Loss: {avg_loss:.4f}\n")
                
                # Commander 专属详细数据打印
                print(f"Solver Success Rate: {p_success:.2%} | "
                      f"Expr Len: {p_len:.2f} | "
                      f"Avg Damage: {p_damage:.2f} | "
                      f"Skill Triggered: {p_skill_trig:.2f} | "
                      f"Skills Used: {p_skill_used:.2f} | "
                      f"Rounds: {p_rounds:.2f} | "
                      f"Turns: {p_turns:.2f} | ")
                # 修复打印格式
                print("Action Dist:", end=" ")
                for k, v in mean_dict.items():
                    print(f"{k}:{v:.2f}", end=" | ")
                print(f'Win DemonV1 Rate: {(p_win_Demon_rate_v1 * 100):.2f}% | '
                      f'Counter DemonV1 Times: {p_counter_Demon_v1}')
                print(f'Win DemonV2 Rate: {(p_win_Demon_rate_v2 * 100):.2f}% | '
                      f'Counter DemonV2 Times: {p_counter_Demon_v2}')
                print(f'Win DemonV3 Rate: {(p_win_Demon_rate_v3 * 100):.2f}% | '
                      f'Counter DemonV3 Times: {p_counter_Demon_v3}')
                print(f'Win DemonV4 Rate: {(p_win_Demon_rate_v4 * 100):.2f}% | '
                      f'Counter DemonV4 Times: {p_counter_Demon_v4}')
                print(f'Win DemonV5 Rate: {(p_win_Demon_rate_v5 * 100):.2f}% | '
                      f'Counter DemonV5 Times: {p_counter_Demon_v5}')

                if self.use_opponent_pool and len(self.opponent_pool) > 0:
                    print("\n--- Opponent Pool Stats ---")
                    print(f"{'Idx':<4} {'WinRate':<8} {'Uses':<6} {'Age':<5} {'Path'}")
                    print("-" * 50)
                    sorted_opponents = sorted(
                        enumerate(self.opponent_pool.opponents), 
                        key=lambda x: x[1]['win_rate'], 
                        reverse=True
                    )
                    for idx, opp in sorted_opponents:
                        print(f"{idx:<4} {opp['win_rate']:<8.2%} {opp['uses']:<6} {opp['age']:<5} {opp['path'].name}")
                    print("---------------------------\n")
                
            if self.episode_num % self.config.get('save_interval', 500) == 0:
                self.save_checkpoint(self.episode_num)
                wandb.save(f"ppo_{self.episode_num}_2.pth")

            if self.episode_num % self.opponent_update_interval == 0 and self.episode_num > 0:
                if self.opponent_pool._compute_diversity_score(self.policy_net.state_dict()) > config.get('diversity_threshold', 0.2):
                    self.opponent_pool.add_opponent(self.policy_net.state_dict(), self.episode_num)
                    print(f"  → Added self to opponent pool")
           
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
        }, f"ppo_{episode}_1.pth")
        print(f"✓ Checkpoint saved at episode {episode}")
    
    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = deque(checkpoint.get('episode_rewards', []), maxlen=100)
        self.losses = deque(checkpoint.get('losses', []), maxlen=100)
        self.episode_num = checkpoint.get('episode', 0)
        print(f"Loaded checkpoint from episode {self.episode_num}")

config = {
        'run_phase': 2, 
        'checkpoint_path': 'ppo_357000_1.pth', 

        'use_opponent_pool': True,
        'opponent_pool_size': 30,
        'opponent_pool_dir': './opponent_pool',
        'opponent_strategy': 'weighted',
        'opponent_update_interval': 1000,
        'min_age_before_elimination': 400,
        'diversity_threshold': 0.2,
        'total_episodes': 600000,  
        'max_episode_steps': 2048,
        'lr': 4e-6,
        'hidden_size': 512,
        'num_block': 8,
        'capacity': 200,
        'ppo_epochs_tot': 4,
        'ppo_epochs_lil': 8,
        'small_update_interval': 20,
        'small_update_recent': 50,
        'mini_batch_size': 4096,
        'clip_param': 0.07,
        'value_coef': 1.5,
        'entropy_coef': 0.03,
        'entropy_decay': 0.99995,
        'entropy_min': 0.02,
        'gamma': 0.999,
        'lam': 0.99,
        'max_grad_norm': 0.75,
        'log_interval': 200,
        'print_interval': 200,
        'print_check_interval': 100,
        'save_interval': 500,
        'adam_eps': 1e-5,
        'reward_scaling': 1.0,
        'weight_decay': 0.005,
        'use_curriculum': False,
        'Main_net_id': 0,
        'curriculum_stage': 1,
        'source_model': None,
        # BC Config
        'bc_coef': 0.02,
        'gold_buffer': 5000,
        'damage_threshold': 0,
        'reward_threshold': 0,
        'counter_demon_prob': 0.6,
        }

if __name__ == "__main__":
    run = wandb.init(
        project="Card-RL-Experiment",
        config=config,
        name=f"Exp_Hier_8MLP_20_2",  
        save_code=True,
        notes="稀疏奖励训练 5代Demon训练 解决过拟合",
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_file_path = os.path.join(script_dir, "Env_sparse.py")
    demonv1_file_path = os.path.join(script_dir, "Demonbot.py")
    demonv2_file_path = os.path.join(script_dir, "Demonbot_2.py")
    demonv3_file_path = os.path.join(script_dir, "Demonbot_3.py")
    demonv4_file_path = os.path.join(script_dir, "Demonbot_4.py")
    demonv5_file_path = os.path.join(script_dir, "Demonbot_5.py")
    wandb.save(env_file_path, base_path=script_dir)
    wandb.save(demonv1_file_path, base_path=script_dir)
    wandb.save(demonv2_file_path, base_path=script_dir)
    wandb.save(demonv3_file_path, base_path=script_dir)
    wandb.save(demonv4_file_path, base_path=script_dir)
    wandb.save(demonv5_file_path, base_path=script_dir)

    env = Game()
    trainer = TrainingManager(env, config)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted manually.")
    finally:
        wandb.finish()