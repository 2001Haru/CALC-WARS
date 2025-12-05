import torch
import numpy as np
from PPO_ver7_experiment_3 import TrainingManager, config as base_config
from Env_exp_3 import Game

def load_and_continue_training(pretrained_path: str, new_config: dict):
    """
    加载预训练模型并继续训练
    """
    #  创建环境和训练管理器
    env = Game()
    trainer = TrainingManager(env, new_config)
    
    #  加载预训练权重
    print(f"Loading pretrained model from {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location=trainer.device, weights_only= False)
    
    # 加载网络参数
    trainer.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    
    # 调整优化器学习率（关键！）
    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] = new_config['lr']  # 使用新学习率
    
    # 调整熵系数
    trainer.entropy_coef = new_config['entropy_coef']
    trainer.trainer.entropy_coef = new_config['entropy_coef']
    
    print(f"✓ Model loaded. Current lr: {trainer.optimizer.param_groups[0]['lr']}")
    print(f"✓ Entropy coef: {trainer.entropy_coef}")
    
    # 5. 继续训练
    trainer.train()
    
    return trainer

if __name__ == "__main__":
    # === 第二阶段训练配置 ===
    # 基于你的base_config，只修改关键参数
    phase2_config = base_config.copy()
    
    # 调整核心超参数
    phase2_config.update({
        # 学习率：降低为原来的1/3~1/5，更稳定
        'lr': 1e-5,  
        'max_episode_steps': 512,
        
        # 熵系数：提高！鼓励探索新策略
        'entropy_coef': 0.03,  
        'entropy_decay': 0.998,  # 衰减
        'entropy_min': 0.002,    # 最低值也提高
        
        # 其他可调参数
        'clip_param': 0.20,      # 保持或略降
        'value_coef': 0.67,       # 提高价值损失权重，稳定训练
        'value_coef': 0.8,

        #GAE计算参数，更注重长序列
        'gamma': 0.996,
        'lam': 0.97,
        
        'mini_batch_size': 64,

        # 训练轮次
        'total_episodes': 10000,  # 训练量
        
        # 保存设置
        'save_interval': 1000,
        'reward_scaling': 100.0,
    })
    
    # 执行训练
    load_and_continue_training(
        pretrained_path='ppo_2021_2.pth',
        new_config=phase2_config
    )