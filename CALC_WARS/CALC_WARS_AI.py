import pygame
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from enum import Enum
from typing import List, Dict, Tuple, Optional
import os
os.environ['AI_DEBUG'] = '0'  # 开启AI调试模式
import numpy as np
from AI_opponents.Smart_solver import FastTemplateSolver


# ==========================================
# 1. 神经网络定义 (CommanderNet)
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(size, size),
            nn.LayerNorm(size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(size, size),
            nn.LayerNorm(size)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.net(x)
        out += identity
        return self.relu(out)

class CommanderNet(nn.Module):
    def __init__(self, obs_dim=92, action_dim=62, hidden_size=512, num_blocks=4):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        self.res_blocks = nn.ModuleList([ResBlock(hidden_size) for _ in range(num_blocks)])
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

    def forward(self, x):
        if len(x.shape) == 1: x = x.unsqueeze(0)
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        return self.policy_head(x), self.value_head(x)


# 初始化pygame
pygame.init()

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
GREEN = (50, 255, 75)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (64, 64, 64)
GOLD_LIGHT = (255, 240, 200)
GOLD_CLASSIC = (255, 215, 0)
GOLD_WARM = (255, 200, 100)
WHITE_CREAM = (255, 250, 240)    # 奶油白，比纯白柔和
WHITE_IVORY = (255, 255, 240)    # 象牙白，略带暖调
WHITE_PURE = (255, 255, 255)
ORANGE_PEACH = (255, 200, 150)   # 桃橙色，温暖醒目
ORANGE_LIGHT = (255, 180, 100)  # 浅橙色，活力但不刺眼
CORAL_SOFT = (255, 150, 120)    # 柔和珊瑚色 
SILVER_LIGHT = (220, 220, 220)   # 亮银色，低调优雅
SILVER_WARM = (230, 230, 210)    # 暖银色，比纯灰温和
PLATINUM = (200, 200, 200)    
BLUE_SKY = (180, 220, 255)       # 天空蓝，清新对比
BLUE_PASTEL = (200, 230, 255)    # 粉蓝色，柔和优雅
BLUE_ELECTRIC = (0, 150, 255)    # 电光蓝，清新醒目
BLUE_NEON = (0, 200, 255)        # 霓虹蓝，明亮对比
BLUE_ROYAL = (0, 100, 255)       # 皇家蓝，深沉醒目
CYAN_LIGHT = (180, 240, 240)     # 浅青色，现代感
YELLOW_LIGHT = (255, 255, 150)   # 浅黄色，明亮柔和
YELLOW_PASTEL = (255, 255, 180)  # 粉黄色，非常温和
YELLOW_CREAM = (255, 245, 180)   # 奶油黄，优雅醒目
ORANGE_NEON = (255, 100, 0)      # 霓虹橙，极度醒目
ORANGE_SUNSET = (255, 120, 40)   # 日落橙，温暖明亮
ORANGE_SAFETY = (255, 150, 50)   # 安全橙，工业级醒目
ORANGE_HOT = (255, 80, 20)       # 热橙色，强烈对比
ORANGE_PUMPKIN = (255, 117, 24)    # 南瓜橙，万圣节风格
RED_NEON = (255, 20, 20)         # 霓虹红，极度醒目
RED_FIRE = (255, 40, 0)          # 火焰红，热烈醒目
RED_CORAL = (255, 80, 60)        # 珊瑚红，柔和但醒目
RED_ELECTRIC = (255, 0, 60)      # 电光红，现代感
YELLOW_ELECTRIC = (255, 230, 0)  # 电光黄，稍柔和
YELLOW_SUN = (255, 200, 0)       # 阳光黄，温暖醒目
YELLOW_SAFETY = (255, 255, 100)  # 安全黄，柔和但清晰
YELLOW_BRIGHT = (255, 255, 150) # 明亮黄，非常醒目

# 屏幕设置
SCREEN_WIDTH = 1300
SCREEN_HEIGHT = 900
CARD_WIDTH = 30
CARD_HEIGHT = 40
CARDS_PER_ROW = 14  # 每行最多显示的卡牌数量

class CardType(Enum):
    NUMBER = "number"
    OPERATOR = "operator"
    SKILL = "skill"

class SkillType(Enum):
    HEAL = "HE"          # 生命恢复牌
    STEAL = "ST"        # 盗窃牌
    DRAW = "DR"          # 抽牌
    SHIELD = "SH"      # 护盾牌
    RUIN = 'RU'   #0牌
    PIERCE = 'PI' #1牌

class Card:
    def __init__(self, value: str, card_type: CardType, skill_type: Optional[SkillType] = None):
        self.value = value
        self.card_type = card_type
        self.skill_type = skill_type
        self.used = False
        self.id = id(self) # 唯一标识符用于区分相同值的卡牌
    
    def __str__(self):
        return self.value

class Player:
    def __init__(self, name: str):
        self.name = name
        self.hp = 120
        self.hand = []
        self.skill_cards = []
        self.shield_count = 0
        self.is_active = False
        self.vec_hand = [0] * 20
        self.vec_skill = [0] * 6
    
    def add_card(self, card: Card):
        if card.card_type == CardType.SKILL:
            self.skill_cards.append(card)
        else:
            self.hand.append(card)
    
    def remove_card(self, card: Card):
        if card in self.hand:
            self.hand.remove(card)
        elif card in self.skill_cards:
            self.skill_cards.remove(card)
    
    def take_damage(self, damage: int):
        if self.shield_count > 0 and damage > 0:
            self.shield_count -= 1
            return 0  # 护盾抵挡伤害
        else:
            self.hp -= damage
            return damage

class Target:
    def __init__(self):
        self.red_zone = random.choice([37,41,43,47,53])
        self.yellow_zone = random.sample([26,27,28,29,30,31,32,33,34,35],2)
        self.blue_zone = random.sample([1,2,3,5,6,7,8,10,11,12,13,14,15,17,18,19,20,21,22,23],3)+ random.sample([4,9,16],1)
    
    def get_damage(self, result: int) -> int:
        if result == self.red_zone:
            return 50
        elif result in self.yellow_zone:
            return 30
        elif result in self.blue_zone:
            return 10
        return 0
    
class AI_Agent:
    def __init__(self):
        self.device = torch.device("cpu") # 推理使用 CPU 即可
        self.net = CommanderNet(obs_dim=92, action_dim=62, num_blocks=8).to(self.device)
        self.solver = FastTemplateSolver()
        self.is_loaded = False
        
        # 映射字典：从字符串到训练时的索引
        self.str_to_idx = {
            '+': 14, '-': 15, '*': 16, '/': 17, '(': 18, ')': 19
        }
        self.skill_map = {
            SkillType.HEAL: 0, SkillType.STEAL: 1, SkillType.DRAW: 2,
            SkillType.SHIELD: 3, SkillType.RUIN: 4, SkillType.PIERCE: 5
        }
    
    def load_model(self, path):
        try:
            if not os.path.exists(path):
                print(f"[AI] Error: Model file {path} not found.")
                return False

            # 首先用默认方式尝试加载（适配多数场景）
            try:
                checkpoint = torch.load(path, map_location=self.device)
            except Exception as e:
                # 处理 PyTorch 2.6+ 的 weights_only 安全加载错误
                err_s = str(e)
                if 'WeightsUnpickler' in err_s or 'weights_only' in err_s or 'Unsupported global' in err_s:
                    print(f"[AI] Model load raised weights-only compatibility error: {e}")
                    try:
                        # 尝试使用 weights_only=False 以保持向后兼容（仅当文件可信时）
                        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                    except Exception as e2:
                        print(f"[AI] Retry with weights_only=False also failed: {e2}")
                        return False
                else:
                    print(f"[AI] Failed to load model: {e}")
                    return False

            # 兼容保存时不同的 key 结构
            state_dict = checkpoint.get('policy_net_state_dict', checkpoint)
            try:
                self.net.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"[AI] Loading state dict failed: {e}")
                return False

            self.net.eval()
            self.is_loaded = True
            print(f"[AI] Model loaded from {path}")
            return True
        except Exception as e:
            print(f"[AI] Failed to load model: {e}")
            return False

    def get_action(self, game_obj) -> int:
        # 如果模型未加载，尽量选择一个合法的保底动作（优先 End Round，再尝试 Solver）
        if not self.is_loaded:
            fallback_mask = self.construct_mask(game_obj)
            if os.environ.get('AI_DEBUG') == '1':
                print(f"[AI_DEBUG] Model not loaded -> fallback_mask={fallback_mask}")
            # 尝试 End Round 优先
            if fallback_mask[61] == 1.0:
                return 61
            # 尝试第一个可行的 Solver 动作
            solver_candidates = [i for i in range(0,54) if fallback_mask[i] == 1.0]
            if solver_candidates:
                return solver_candidates[0]
            # 最后退化为 End Turn
            return 60 # 没模型就 End Turn（兜底）

        # 1. 构造 State Vector (92 dim)
        state_vec = self.construct_state_vector(game_obj)

        # 2. 构造 Mask
        mask = self.construct_mask(game_obj)
        # 保证至少有一个可行动作（与 Env 中逻辑保持一致）
        if mask.sum() == 0:
            mask[61] = 1.0  # End Round 作为最后保底
            if os.environ.get('AI_DEBUG') == '1':
                print(f"[AI_DEBUG] mask all zero -> force End Round (mask set to 61)")

        # 3. 推理（更鲁棒的掩码/数值处理）
        state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
        mask_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, _ = self.net(state_t)
            # 屏蔽不可行动作
            logits = logits.masked_fill(~mask_t, float('-inf'))
            # 防止全 -inf 导致 softmax NaN，clamp 到大负数
            logits = torch.clamp(logits, min=-1e5, max=1e3)
            # 额外去 NaN/Inf
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                if os.environ.get('AI_DEBUG') == '1':
                    print(f"[AI_DEBUG] logits contained NaN/Inf. mask={mask}, logits(before)={logits}")
                logits = torch.nan_to_num(logits, nan=-1e5, posinf=1e3, neginf=-1e5)
                if os.environ.get('AI_DEBUG') == '1':
                    print(f"[AI_DEBUG] logits sanitized -> logits(after)={logits}")

            # 更稳健的选取：在 logits 层进行 masked-argmax，而不是依赖 probs
            logits_sanitized = logits.clone()
            neg_inf_fill = torch.tensor(-1e6, device=logits.device)
            logits_sanitized = torch.where(mask_t, logits_sanitized, neg_inf_fill)

            # 选取最大 logit 对应动作
            action_tensor = logits_sanitized.argmax(dim=-1)
            action = int(action_tensor.item())

            # Debug 信息
            if os.environ.get('AI_DEBUG') == '1':
                valid_indices = torch.where(mask_t[0])[0].cpu().numpy().tolist()
                top_logits = [(int(i), float(logits[0,i].item())) for i in valid_indices]
                print(f"[AI_DEBUG] mask={mask} valid_idxs={valid_indices} top_logits={top_logits} chosen={action}")

            # 如果所选动作被认为非法（保险），回退到有效掩码下的最大值
            if not mask[int(action)]:
                # fallback to masked logits argmax
                valid_logits = logits_sanitized[0]
                action = int(torch.argmax(valid_logits).item())
                if os.environ.get('AI_DEBUG') == '1':
                    print(f"[AI_DEBUG] fallback to masked argmax chosen={action}")

            # 最终兜底：若仍然不合法，则选择 End Round/End Turn
            if not bool(mask[int(action)]):
                if mask[61] == 1.0:
                    return 61
                if mask[60] == 1.0:
                    return 60
                return 61

        return int(action)

    def construct_state_vector(self, game) -> np.ndarray:
        # 必须完全复刻 Env_sparse.py 的 _get_state 逻辑
        vec = np.zeros(92, dtype=np.float32)
        
        # 辅助函数：将 player.hand (List[Card]) 转为计数数组 (size 20)
        def get_hand_counts(hand_list):
            counts = [0] * 20
            for card in hand_list:
                if card.card_type == CardType.NUMBER:
                    val = int(card.value)
                    if 0 <= val <= 13: counts[val] += 1
                elif card.card_type == CardType.OPERATOR:
                    idx = self.str_to_idx.get(card.value)
                    if idx: counts[idx] += 1
            return np.array(counts)

        def get_skill_counts(skill_list):
            counts = [0] * 6
            for card in skill_list:
                if card.skill_type in self.skill_map:
                    counts[self.skill_map[card.skill_type]] += 1
            return np.array(counts)

        # AI是Player2
        ai_p = game.player2
        hu_p = game.player1
        
        # 1. 手牌 (归一化 / 5.0)
        ai_hand_cnt = get_hand_counts(ai_p.hand)
        hu_hand_cnt = get_hand_counts(hu_p.hand)
        vec[:20] = ai_hand_cnt / 5.0
        vec[20:40] = hu_hand_cnt / 5.0
        
        # 2. 技能 (归一化 / 3.0)
        ai_skill_cnt = get_skill_counts(ai_p.skill_cards)
        hu_skill_cnt = get_skill_counts(hu_p.skill_cards)
        vec[40:46] = ai_skill_cnt / 3.0
        vec[46:52] = hu_skill_cnt / 3.0
        
        # 3. 核心信息
        vec[52] = ai_p.hp / 120.0
        vec[53] = hu_p.hp / 120.0
        vec[54] = ai_p.shield_count / 4.0
        vec[55] = hu_p.shield_count / 4.0
        vec[56] = game.round_number / 5.0
        vec[57] = game.turn_number / 10.0   # 待实现
        vec[58] = game.step_count / 100.0
        vec[59] = game.continuous_operations / 5.0
        
        # 62-64 Flow
        # 4. 差距信息
        # 注意: 这里用的是归一化后的 sum，与 Env 保持一致
        vec[60] = (vec[:20].sum() - vec[20:40].sum()) / 10.0 # 手牌差距 (归一化值之差)
        vec[61] = (ai_p.hp - hu_p.hp) / 50.0
        
        # 5. Flow 控制位 (62-64)
        if game.first_to_end_round == ai_p: vec[62] = 1.0
        elif game.first_to_end_round == hu_p: vec[62] = -1.0
        else: vec[62] = 0.0
        
        if game.current_player == game.player1:
            vec[63] = 1.0 if game.player1_round_end else 0.0
            vec[64] = 1.0 if game.player2_round_end else 0.0
        else:
            vec[63] = 1.0 if game.player2_round_end else 0.0
            vec[64] = 1.0 if game.player1_round_end else 0.0
    
        
        # 5. 目标 (归一化 / 40.0)
        t = game.target
        zones = [t.red_zone] + t.yellow_zone[:2] + t.blue_zone[:4]
        # 补齐到7位
        while len(zones) < 7: zones.append(0)
        vec[65:72] = np.array(zones[:7], dtype=np.float32) / 40.0
        
        # 高级特征提取
        # 7. 高级特征提取 (Extract Features) - 严格复刻
        extract_feature = np.zeros(18, dtype=np.float32)
        
        # 资源结构分析
        # 需要还原 hand list 的扁平结构: [0,0,0, 14, 14, ...]
        # 在 Env_sparse 中是直接根据 hand array (count) 展开
        nums = []
        ops = []
        # AI 手牌计数数组已经有 ai_hand_cnt (对应 index 0-19)
        for i in range(14): 
            nums.extend([i] * int(ai_hand_cnt[i]))
        for i in range(14, 18): # 不含括号
            ops.extend([i] * int(ai_hand_cnt[i]))
        
        has_bracket = (ai_hand_cnt[18] > 0 and ai_hand_cnt[19] > 0)
        
        num_cnt = len(nums)
        op_cnt = len(ops)
        
        # 特征 0-2: 资源结构
        extract_feature[0] = num_cnt / 10.0
        extract_feature[1] = op_cnt / 5.0
        extract_feature[2] = op_cnt / (num_cnt + 0.1) # 卡手指数
        
        # 特征 3-6: 蓝色区域直接匹配
        # zones[3:7] 是 blue zone
        extract_feature[3:7] = [1.0 if zones[i+3] in nums else 0.0 for i in range(4)]
        
        # 特征 7-13: 技能数字直接匹配
        skill_targets = [0, 1, 2, 4, 6, 8, 9]
        extract_feature[7:14] = [1.0 if skill_targets[i] in nums else 0.0 for i in range(7)]
        
        # 特征 14: 括号
        extract_feature[14] = 1.0 if has_bracket else 0.0
        
        # 特征 15: 偷牌收益 (手牌差 < 6)
        # 注意: 这里用的是归一化后的 sum 比较
        extract_feature[15] = 1.0 if (vec[:20].sum() - vec[20:40].sum()) <= 6.0 / 5.0 else 0.0
        
        # 特征 16: Pierce 需求 (对手盾 >= 2)
        extract_feature[16] = 1.0 if hu_p.shield_count >= 2.0 else 0.0
        
        # 特征 17: 斩杀提示 (对手血 <= 50)
        extract_feature[17] = 1.0 if hu_p.hp <= 50.0 else 0.0
        
        # 填入 vec 72-89
        vec[72:90] = extract_feature[:18]

        # 游戏结束标志
        vec[91] = 0.0
        
        return vec

    def construct_mask(self, game) -> np.ndarray:
        mask = np.zeros(62, dtype=np.float32)
        ai_p = game.player2
        
        # 1. Skills (54-59)
        skill_cnt = [0]*6
        for c in ai_p.skill_cards:
            if c.skill_type in self.skill_map:
                skill_cnt[self.skill_map[c.skill_type]] += 1
        mask[54:60] = (np.array(skill_cnt) > 0).astype(np.float32)
        
        # 2. End Turn (60) / Round (61)
        mask[60] = 1.0
        if game.player1_round_end or game.player2_round_end:
            mask[60] = 0.0 # 只要有一方结束本轮，就不能再End Turn，只能End Round
        mask[61] = 1.0
        
        # 3. Solver (0-53)
        # 转换手牌格式给 Solver
        hand_counts = {}
        for card in ai_p.hand:
            idx = -1
            if card.card_type == CardType.NUMBER: idx = int(card.value)
            elif card.card_type == CardType.OPERATOR: idx = self.str_to_idx.get(card.value)
            
            if idx != -1:
                hand_counts[idx] = hand_counts.get(idx, 0) + 1
                
        # 调用 Solver 获得可达集
        r3, r5 = self.solver.get_reachable_sets(hand_counts)
        
        # 目标映射
        t = game.target
        zones = [t.red_zone] + t.yellow_zone + t.blue_zone
        spec_map = {0:2, 1:6, 2:120, 3:8, 4:27, 5:64, 
                    6:4, 7:9, 8:16, 9:25, 10:36, 11:49, 12:81, 13:100, 14:121, 15:144, 16:169, 
                    17:24, 18:0, 19:1}
        
        for i in range(27):
            target = None
            if i < 7: 
                if i < len(zones): target = zones[i]
            else: target = spec_map.get(i - 7)
            
            if target is None: continue
            
            is_eco = target in r3
            is_std = is_eco or (target in r5)
            
            base_idx = 2 * i if i < 7 else 14 + 2 * (i - 7)
            if is_eco: mask[base_idx] = 1.0
            if is_std: mask[base_idx + 1] = 1.0
            
        if mask.sum() == 0: mask[61] = 1.0
        return mask
    
class Button:
    def __init__(self, x, y, width, height, text, color, hover_color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text

        # Windows XP经典按钮颜色
        self.normal_color = (225, 225, 225)    # 浅灰色按钮底色
        self.hover_color = (195, 195, 195)     # 悬停时稍深的灰色
        self.border_light = (255, 255, 255)     # 亮边框
        self.border_dark = (128, 128, 128)     # 暗边框
        self.text_color = (0, 0, 0)            # 黑色文字
        
        # 允许自定义颜色，如果不传则使用默认
        if color:
            self.normal_color = color
        if hover_color:
            self.hover_color = hover_color
            
        self.is_hovered = False
        self.is_pressed = False
        
    def draw(self, surface):
        # 确定按钮底色
        if self.is_pressed:
            base_color = (170, 170, 170)  # 按下时更深
            text_offset = 2  # 按下时文字偏移
        else:
            base_color = self.hover_color if self.is_hovered else self.normal_color
            text_offset = 0

        # 绘制按钮主体
        pygame.draw.rect(surface, base_color, self.rect)
        

        if self.is_pressed:
            # 按下状态：暗边框在外，亮边框在内
            pygame.draw.line(surface, self.border_dark, self.rect.topleft, self.rect.topright, 2)
            pygame.draw.line(surface, self.border_dark, self.rect.topleft, self.rect.bottomleft, 2)
            pygame.draw.line(surface, self.border_light, (self.rect.left+1, self.rect.bottom-1), 
                           (self.rect.right-1, self.rect.bottom-1), 2)
            pygame.draw.line(surface, self.border_light, (self.rect.right-1, self.rect.top+1), 
                           (self.rect.right-1, self.rect.bottom-1), 2)
        else:
            # 正常状态：亮边框在外，暗边框在内
            pygame.draw.line(surface, self.border_light, self.rect.topleft, self.rect.topright, 2)
            pygame.draw.line(surface, self.border_light, self.rect.topleft, self.rect.bottomleft, 2)
            pygame.draw.line(surface, self.border_dark, (self.rect.left+1, self.rect.bottom-1), 
                           (self.rect.right-1, self.rect.bottom-1), 2)
            pygame.draw.line(surface, self.border_dark, (self.rect.right-1, self.rect.top+1), 
                           (self.rect.right-1, self.rect.bottom-1), 2)
        
        # 绘制文字 按下状态文字稍微偏移，模拟按下效果
        # 避免依赖模块级的 `game` 变量（在被外部 import 时可能未定义），使用本地字体渲染
        text_surface = pygame.font.SysFont("times new roman", 24).render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=(self.rect.centerx + text_offset, 
                                                self.rect.centery + text_offset))
        surface.blit(text_surface, text_rect)
        
    def check_hover(self, mouse_pos):
        #检查鼠标是否悬停在按钮上
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        return self.is_hovered
        
    def is_clicked(self, mouse_pos, mouse_click):
        #检查按钮是否被点击
        if self.rect.collidepoint(mouse_pos) and mouse_click:
            self.is_pressed = True
            return True
        return False
    
    def update_press_state(self, mouse_buttons):
        #更新按钮按下状态 需要在鼠标释放时调用
        if not mouse_buttons[0]:  # 左键释放
            self.is_pressed = False
    

class Game:
    def _init_buttons(self):
            """初始化所有按钮"""
            # 主界面按钮 - 使用Windows XP经典按钮样式
            self.buttons['skill'] = Button(50, 265, 150, 50, "Skill Cards", 
                                        (255, 255, 100), (255, 255, 150))  # 黄色系
            self.buttons['confirm'] = Button(50, 330, 150, 50, "Confirm", 
                                            (100, 255, 100), (150, 255, 150))  # 绿色系
            self.buttons['end'] = Button(50, 390, 150, 50, "End", 
                                        (100, 100, 255), (150, 150, 255))  # 蓝色系
            self.buttons['end_round'] = Button(50, 460, 150, 50, "End the Round", 
                                            (255, 100, 100), (255, 150, 150))  # 红色系

    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("24 Points Cards Game")
        self.clock = pygame.time.Clock()
        # 加载支持中文的字体
        self.font = self.load_font(24)
        self.small_font = self.load_font(20)
        self.card_font = self.load_font(20)  # 卡牌专用字体
        self.large_font = self.load_font(41)
        self.super_large_font = self.load_font(80)
        
        self.player1 = Player("Player1")
        self.player2 = Player("Player2")
        self.current_player = self.player1
        self.target = Target()
        self.game_phase = "initial_deal"  # initial_deal, playing, round_end
        self.round_number = 1
        self.turn_number = 1
        self.selected_cards = []
        self.calculation_result = None
        self.message = ""
        self.history = []  # 历史记录
        self.action_messages = []  # 操作消息队列
        self.player1_round_end = False  # 玩家1是否结束本轮
        self.player2_round_end = False  # 玩家2是否结束本轮
        self.continuous_operations = 0  # 连续操作次数
        self.first_to_end_round = None  # 当前轮次先结束本轮的玩家
        self.step_count = 0

        self.calculation_display_time = 0  # 记录结果显示时间
        self.result_alpha = 255  # 透明度(255为完全不透明)
        self.showing_result = False  # 是否正在显示结果
        self.fade_duration = 1500  # 淡出持续时间(毫秒)
        self.broadcastmessage = ""
        # 广播消息显示相关
        self.broadcast_display_time = 0  # 广播消息显示时间
        self.broadcast_alpha = 255  # 广播消息透明度
        self.showing_broadcast = False  # 是否正在显示广播消息

        # AI 相关状态
        self.agent = AI_Agent()
        # 若外部传入 model_path，则加载（Menu 会在实例化后再加载正确模型）
        if hasattr(self, 'model_path') and self.model_path:
            self.agent.load_model(self.model_path)
        self.ai_thinking_start = 0
        self.is_ai_thinking = False

        # 初始化音乐
        pygame.mixer.init()  # 初始化混音器

        self.music_list = [
            "sounds/background.ogg",  
            "sounds/background1.ogg", 
            'sounds/background2.ogg'
        ]

        self.current_music_index = 0

        self.background_music = pygame.mixer.Sound('sounds/background.ogg')  # 加载背景音乐
        self.sound_effects = {
            'click': pygame.mixer.Sound('sounds/click.wav'),
            'skill_trigger': pygame.mixer.Sound('sounds/skill_trigger.wav'),
            'damage': pygame.mixer.Sound('sounds/damage.wav'),
            'shield_block': pygame.mixer.Sound('sounds/shield_block.wav'),
            'heal': pygame.mixer.Sound('sounds/heal.wav'),
            'pierce': pygame.mixer.Sound('sounds/pierce.wav'),
            'steal': pygame.mixer.Sound('sounds/steal.wav'),
            'ruin': pygame.mixer.Sound('sounds/ruin.wav'),
            'draw': pygame.mixer.Sound('sounds/draw.wav'),
            'shield': pygame.mixer.Sound('sounds/shield.wav'),
            'win': pygame.mixer.Sound('sounds/win.wav')
        }
        
        # 音乐音量设置
        self.music_volume = 0.5  # 0.0到1.0
        self.sfx_volume = 0.7
        
        self.buttons = {}
        
        self._init_buttons()

        self.show_rules = False
        
        # 规则文本
        self.rules_text  = [
            "Factorial: One Heal Card( Recover 20 HP and draw one card )",
            "Cube: One Steal Card( Steal 2 cards randomly )",
            "Square: One Draw Card( Draw 2 numbers and 1 operator )",
            "24 Points: Two Shield Cards( Get 1 shield )",
            "ZERO: One Ruin card( Destroy 3 cards ramdomly )",
            "ONE: One Pierce card( Break the opponent's shield )",
            '',
            'If the difference in hand cards between the two players is 6 or more',
            'Stealing only steals 1 card!!'
        ]


        # 初始化游戏
        self.initial_deal()

    def play_background_music(self, loop=-1):
        """播放背景音乐"""
        self.background_music.set_volume(self.music_volume)
        self.background_music.play(loop)  # -1表示无限循环

    def play_next_music(self):
        """播放下一首音乐"""
        if not self.music_list:  # 如果列表为空
            return
            
        # 加载并播放当前音乐
        try:
            pygame.mixer.music.load(self.music_list[self.current_music_index])
            pygame.mixer.music.play()
            print(f"正在播放: {self.music_list[self.current_music_index]}")
        except pygame.error as e:
            print(f"无法播放音乐: {e}")
            
        # 移动到下一首（循环）
        self.current_music_index = (self.current_music_index + 1) % len(self.music_list)

    def stop_background_music(self):
        """停止背景音乐"""
        self.background_music.stop()

    def play_sound_effect(self, effect_name):
        """播放音效"""
        if effect_name in self.sound_effects:
            self.sound_effects[effect_name].set_volume(self.sfx_volume)
            self.sound_effects[effect_name].play()

    def set_music_volume(self, volume):
        """设置音乐音量"""
        self.music_volume = max(0.0, min(1.0, volume))
        self.background_music.set_volume(self.music_volume)

    def set_sfx_volume(self, volume):
        """设置音效音量"""
        self.sfx_volume = max(0.0, min(1.0, volume))
        for sound in self.sound_effects.values():
            sound.set_volume(self.sfx_volume)

    def load_font(self, size):
        try:
            return pygame.font.SysFont("times new roman", size)
        except:
            return pygame.font.SysFont()
    
    def get_card_position(self, index: int, start_x: int, start_y: int) -> Tuple[int, int]:
        """计算卡牌位置，支持自动换行"""
        row = index // CARDS_PER_ROW
        col = index % CARDS_PER_ROW
        x = start_x + col * (CARD_WIDTH + 5)
        y = start_y + row * (CARD_HEIGHT + 5)
        return x, y
    
    def add_to_history(self, player_name: str, expression: str, result: int, damage: int, skill_triggered: str = ""):
        """添加计算结果到历史记录"""
        history_entry = {
            "player": player_name,
            "expression": expression,
            "result": result,
            "damage": damage,
            "skill": skill_triggered,
            "round": self.round_number
        }
        self.history.append(history_entry)
    
    def add_action_message(self, message: str):
        """添加操作消息"""
        self.action_messages.append(message)
        # 限制消息数量，避免界面过于拥挤
        if len(self.action_messages) > 5:
            self.action_messages.pop(0)
    
    def initial_deal(self):
        """初始发牌"""
        for _ in range(5):
            self.player1.add_card(self.generate_random_number_card())
            self.player2.add_card(self.generate_random_number_card())
        for _ in range(2):
            self.player1.add_card(self.generate_random_op_card())
            self.player2.add_card(self.generate_random_op_card())
        
        # 随机决定先手
        if random.random() < 0.5:
            self.current_player = self.player1
        else:
            self.current_player = self.player2

        # 后手补偿RUIN
        if self.current_player == self.player1:
            self.player2.add_card(self.generate_skill_card(SkillType.RUIN))
        else:
            self.player1.add_card(self.generate_skill_card(SkillType.RUIN))
        
        self.current_player.is_active = True
        self.game_phase = "playing"
        self.message = f"{self.current_player.name} Starts the Round"
        self.player1_round_end = False
        self.player2_round_end = False
        self.continuous_operations = 0
    
    def generate_random_card(self) -> Card:
        """生成随机卡牌"""
        if random.random() < 0.66667:  # 66.667%概率数字牌
            return Card(str(random.randint(0, 13)), CardType.NUMBER)
        else:  # 33.333%概率符号牌
            operators = ['+','+','-','-', '*', '*','/','+','-','*','+','+','-','-', '*', '*','/','/','/','+','-','*','(', ')'] 
            #加号概率：减号概率：乘号概率：除号概率：括号概率=6:6:6:4:2
            return Card(random.choice(operators), CardType.OPERATOR)
        
    def generate_random_number_card(self) -> Card:
        """生成随机数字牌"""
        return Card(str(random.randint(0, 13)), CardType.NUMBER)
    
    def generate_random_op_card(self) ->Card:
        """生成随机符号牌"""
        # 特化发牌，不发括号
        operators = ['+','-', '*','+','-', '*','+','-', '*','/','/'] 
        #加号概率：减号概率：乘号概率：除号概率=1:1:1:1
        return Card(random.choice(operators), CardType.OPERATOR)
    
    def generate_skill_card(self, skill_type: SkillType) -> Card:
        """生成技能牌"""
        return Card(skill_type.value, CardType.SKILL, skill_type)
    
    def calculate_expression(self, cards: List[Card]) -> Optional[int]:
        """计算表达式结果"""
        try:
            for i,card in enumerate(cards):
                if i <= len(cards) - 2:
                    if cards[i].card_type == CardType.NUMBER and \
                        cards[i+1].card_type == CardType.NUMBER:
                        return None
                    
            # 构建表达式字符串
            expression = ""
            for card in cards:
                expression += card.value
            
            # 检查是否包含运算符 禁止直接组合数字
            if not any(op in expression for op in ['+', '-', '*', '/', '(', ')']):
                return None
            
            # 安全的表达式计算
            if self.is_valid_expression(expression):
                # 使用更安全的计算方式
                allowed_chars = set('0123456789+-*/().')
                if all(c in allowed_chars for c in expression):
                    result = eval(expression)
                    if isinstance(result, (int, float)) and not math.isnan(result) and not math.isinf(result):
                        if int(result) == result:
                            return int(result)
                        else:
                            return round(result,2)
        except (ZeroDivisionError, SyntaxError, ValueError):
            pass
        return None
    
    def is_valid_expression(self, expr: str) -> bool:
        """检查表达式是否有效"""
        # 简单检查：不能以运算符开头或结尾
        if expr[0] in '+-*/)' or expr[-1] in '+-*/(':
            return False
        return True
    
    def check_skill_triggers(self, result: int) -> str:
        self.broadcastmessage = ""
        """检查技能触发条件，返回触发的技能描述"""
        if result is None:
            return ""
        
        skill_message = ""
        #检查1
        if result == 1:
            self.current_player.add_card(self.generate_skill_card(SkillType.PIERCE))
            skill_message = f" ONE skill triggered! Got pierce card"
            self.add_action_message(f"{self.current_player.name} calculated {result}! ONE skill triggered!")
            self.broadcastmessage = f"{self.current_player.name} triggered ONE skill and got a pierce card!"
            self.play_sound_effect('skill_trigger')
            # 添加以下三行
            self.broadcast_display_time = pygame.time.get_ticks()
            self.broadcast_alpha = 255
            self.showing_broadcast = True

        #检查0
        if result ==0:
            self.current_player.add_card(self.generate_skill_card(SkillType.RUIN))
            skill_message = f" ZERO skill triggered! Got ruin card"
            self.add_action_message(f"{self.current_player.name} calculated {result}! ZERO skill triggered!")
            self.broadcastmessage = f"{self.current_player.name} triggered ZERO skill and got a ruin card!"
            self.play_sound_effect('skill_trigger')
            # 添加以下三行
            self.broadcast_display_time = pygame.time.get_ticks()
            self.broadcast_alpha = 255
            self.showing_broadcast = True

        # 检查阶乘
        for i in range(2, 13):
            if math.factorial(i) == result:
                self.current_player.add_card(self.generate_skill_card(SkillType.HEAL))
                skill_message = f" Factorial skill triggered! Got heal card"
                self.add_action_message(f"{self.current_player.name} calculated the factorial of {i}! Factorial skill triggered!")
                self.broadcastmessage = f"{self.current_player.name} triggered Factorial skill and got a heal card!"
                self.play_sound_effect('skill_trigger')
                # 添加以下三行
                self.broadcast_display_time = pygame.time.get_ticks()
                self.broadcast_alpha = 255
                self.showing_broadcast = True
                break
        
        # 检查cube
        if result > 0:
            cbrt_result = int(round(result ** (1/3)))
        else:
            cbrt_result = 20000
        if cbrt_result ** 3 == result and result!= 0 and result != 1:
            self.current_player.add_card(self.generate_skill_card(SkillType.STEAL))
            skill_message = f" Cube skill triggered! Got steal card"
            self.add_action_message(f"{self.current_player.name} calculated the cube of {cbrt_result}! Cube skill triggered!")
            self.broadcastmessage = f"{self.current_player.name} triggered Cube skill and got a steal card!"
            self.play_sound_effect('skill_trigger')
            # 添加以下三行
            self.broadcast_display_time = pygame.time.get_ticks()
            self.broadcast_alpha = 255
            self.showing_broadcast = True
        
        # 检查square
        if result > 0:
            sqrt_result = int(math.sqrt(result))
        else:
            sqrt_result = 20000
        if sqrt_result ** 2 == result and result != 0 and result != 1:
            self.current_player.add_card(self.generate_skill_card(SkillType.DRAW))
            skill_message = f" Square skill triggered! Got Draw card"
            self.add_action_message(f"{self.current_player.name} calculated the square of {sqrt_result}! Square skill triggered!")
            self.broadcastmessage = f"{self.current_player.name} triggered Square skill and got a draw card!"
            self.play_sound_effect('skill_trigger')
            # 添加以下三行
            self.broadcast_display_time = pygame.time.get_ticks()
            self.broadcast_alpha = 255
            self.showing_broadcast = True
        
        # 检查24点
        if result == 24:
            self.current_player.add_card(self.generate_skill_card(SkillType.SHIELD))
            self.current_player.add_card(self.generate_skill_card(SkillType.SHIELD))
            skill_message = f" 24-point skill triggered! Got 2 shield cards"
            self.add_action_message(f"{self.current_player.name} calculated 24 points! 24-point skill triggered!")
            self.broadcastmessage = f"{self.current_player.name} triggered 24-point skill and got 2 shield cards!"
            self.play_sound_effect('skill_trigger')
            # 添加以下三行
            self.broadcast_display_time = pygame.time.get_ticks()
            self.broadcast_alpha = 255
            self.showing_broadcast = True
        
        return skill_message
    
    def use_skill_card(self, skill_card: Card):
        
        self.broadcastmessage = ""
        """使用技能牌"""
        if skill_card.skill_type == SkillType.HEAL:
            # 恢复20点生命值并抽1张牌
            self.current_player.hp = min(120, self.current_player.hp + 20)
            self.current_player.add_card(self.generate_random_card())
            self.message = f"{self.current_player.name} used heal card, recovered 20 HP and got one card!"
            self.add_action_message(f"{self.current_player.name} used heal card, recovered 20 HP and got one card!")
            self.broadcastmessage = f"{self.current_player.name} used heal card, recovered 20 HP and got one card!"
            self.play_sound_effect('heal')
            # 添加以下三行
            self.broadcast_display_time = pygame.time.get_ticks()
            self.broadcast_alpha = 255
            self.showing_broadcast = True

        elif skill_card.skill_type == SkillType.PIERCE:
            # 直接破坏对方所有护盾
            other_player = self.player2 if self.current_player == self.player1 else self.player1
            other_player.shield_count =0
            self.message = f"{self.current_player.name} used pierce card, broke {other_player.name}'s shield!"
            self.add_action_message(f"{self.current_player.name} used pierce card, broke {other_player.name}'s shield!")
            self.broadcastmessage = f"{self.current_player.name} used pierce card and broke {other_player.name}'s shield!"
            self.play_sound_effect('pierce')
            # 添加以下三行
            self.broadcast_display_time = pygame.time.get_ticks()
            self.broadcast_alpha = 255
            self.showing_broadcast = True
        
        elif skill_card.skill_type == SkillType.RUIN:
            # 随机从对方手牌中毁掉3张
            other_player = self.player2 if self.current_player == self.player1 else self.player1
            #self.current_player.add_card(self.generate_random_card())
            ruinlist = []
            for i in range(3):
                if other_player.hand:
                    ruin_card = random.choice(other_player.hand)
                    ruinlist.append(ruin_card.value)
                    other_player.remove_card(ruin_card)
                else:
                    break

            self.message = f"{self.current_player.name} used ruin card, destroyed {' '.join(ruinlist)}!"
            self.add_action_message(f"{self.current_player.name} used ruin card, destroyed {' '.join(ruinlist)}!")
            self.broadcastmessage = f"{self.current_player.name} used ruin card and destroyed {' '.join(ruinlist)} from {other_player.name}!"
            self.play_sound_effect('ruin')
            # 添加以下三行
            self.broadcast_display_time = pygame.time.get_ticks()
            self.broadcast_alpha = 255
            self.showing_broadcast = True
        elif skill_card.skill_type == SkillType.STEAL:
            # 随机从对方手牌中偷牌，若手牌差距大于等于6张则偷2张，否则偷1张
            other_player = self.player2 if self.current_player == self.player1 else self.player1
            stolelist = []
            if self.current_player.hand.__len__() - other_player.hand.__len__() <=6:
                for i in range(2):
                    if other_player.hand:  
                        stolen_card = random.choice(other_player.hand)
                        other_player.remove_card(stolen_card)
                        self.current_player.add_card(stolen_card)
                        stolelist.append(stolen_card.value)
                    else:
                        break
            else:
                if other_player.hand:
                    stolen_card = random.choice(other_player.hand)
                    other_player.remove_card(stolen_card)
                    self.current_player.add_card(stolen_card)
                    stolelist.append(stolen_card.value)

            self.message = f"{self.current_player.name} used steal card, stole {' '.join(stolelist)}"
            self.add_action_message(f"{self.current_player.name} used steal card, stole {' '.join(stolelist)}!")
            self.broadcastmessage = f"{self.current_player.name} used steal card and stole {' '.join(stolelist)} from {other_player.name}!"
            self.play_sound_effect('steal')
            # 添加以下三行
            self.broadcast_display_time = pygame.time.get_ticks()
            self.broadcast_alpha = 255
            self.showing_broadcast = True

        elif skill_card.skill_type == SkillType.DRAW:
            drawlist = []
            for _ in range(2):
                self.current_player.add_card(self.generate_random_number_card())
                drawlist.append(self.current_player.hand[-1].value)
            self.current_player.add_card(self.generate_random_op_card())
            drawlist.append(self.current_player.hand[-1].value)
            self.message = f"{self.current_player.name} used draw skill, got {' '.join(drawlist)}!"
            self.add_action_message(f"{self.current_player.name} used draw skill, got {' '.join(drawlist)}!")
            self.broadcastmessage = f"{self.current_player.name} used draw skill and got {' '.join(drawlist)}!"
            self.play_sound_effect('draw')
            # 添加以下三行
            self.broadcast_display_time = pygame.time.get_ticks()
            self.broadcast_alpha = 255
            self.showing_broadcast = True

        elif skill_card.skill_type == SkillType.SHIELD:
            self.current_player.shield_count += 1
            self.message = f"{self.current_player.name} used shield card, got 1 shield"
            self.add_action_message(f"{self.current_player.name} used shield card, got 1 shield!")
            self.broadcastmessage = f"{self.current_player.name} used shield card and got 1 shield!"
            self.play_sound_effect('shield')
            # 添加以下三行
            self.broadcast_display_time = pygame.time.get_ticks()
            self.broadcast_alpha = 255
            self.showing_broadcast = True
        
        self.current_player.remove_card(skill_card)
    
    def switch_player(self):
        """切换玩家 结束一次连续操作"""
        # 切换玩家时 Turn +1
        self.turn_number += 1
        # 操作步骤 +1 (End Turn 也算一步)
        self.step_count += 1
        self.current_player.is_active = False
        
        self.current_player = self.player2 if self.current_player == self.player1 else self.player1
        self.current_player.is_active = True
        self.selected_cards = []
        self.calculation_result = None
        self.continuous_operations = 0
        
        # 检查切换到的玩家是否已结束本轮
        if (self.current_player == self.player1 and self.player1_round_end) or \
           (self.current_player == self.player2 and self.player2_round_end):
            # 如果切换到的玩家已结束本轮，继续切换
            self.switch_player()
        else:
            self.message = f"{self.current_player.name}'s turn"
    
    def end_current_round(self):
        """当前玩家结束本轮"""
        # 结束本轮也是一步操作
        self.step_count += 1
        if self.current_player == self.player1:
            self.player1_round_end = True
        else:
            self.player2_round_end = True
        
        # 记录先结束本轮的玩家
        if self.first_to_end_round is None:
            self.first_to_end_round = self.current_player

         # 添加广播消息
        self.broadcastmessage = f"{self.current_player.name} ended the round!"
        self.broadcast_display_time = pygame.time.get_ticks()
        self.broadcast_alpha = 255
        self.showing_broadcast = True
        self.add_action_message(f"{self.current_player.name} Round Over")
        
        # 检查是否双方都结束了本轮
        if self.player1_round_end and self.player2_round_end:
            self.end_round()
        else:
            # 切换给对手 如果对手还未结束本轮
            other_player = self.player2 if self.current_player == self.player1 else self.player1
            if (other_player == self.player1 and not self.player1_round_end) or \
               (other_player == self.player2 and not self.player2_round_end):
                self.current_player.is_active = False
                self.current_player = other_player
                self.current_player.is_active = True
                self.selected_cards = []
                self.calculation_result = None
                self.continuous_operations = 0
                self.message = f"{self.current_player.name}'s turn"
                # 切换玩家时 Turn +1
                self.turn_number += 1
            else:
                # 对手也已结束本轮，直接结束轮次
                self.end_round()
    
    def end_round(self):
        """结束当前轮次 双方都结束本轮后调用"""
        # 下一轮先手权给当前轮次先结束本轮的玩家
        # 结束轮次也是一步操作（逻辑上）
        self.turn_number += 1
        if self.first_to_end_round is not None:
            self.current_player = self.first_to_end_round
            if self.current_player == self.player1:
                self.player2.add_card(self.generate_random_card())
                self.player2.add_card(self.generate_random_card())
            else:
                self.player1.add_card(self.generate_random_card())
                self.player1.add_card(self.generate_random_card())
        else:
            # 如果同时结束，保持当前玩家先手
            pass
        
        self.current_player.is_active = True
        
        # 发新牌
        self.newcardsnum = 5 + int(self.round_number /2)  # 每2轮增加1张新牌
        for _ in range(self.newcardsnum):
            self.player1.add_card(self.generate_random_card())
            self.player2.add_card(self.generate_random_card())
        
        self.round_number += 1
        self.selected_cards = []
        self.calculation_result = None
        self.player1_round_end = False
        self.player2_round_end = False
        self.continuous_operations = 0
        self.first_to_end_round = None  # 重置先结束本轮的玩家记录
        
        # 清空历史消息和操作消息
        self.history = []
        self.action_messages = []
        
        self.message = f"Round {self.round_number} starts, {self.current_player.name} goes first"

    def execute_ai_turn(self):
        # 1. 检查是否正在“思考” (UI延迟)
        current_time = pygame.time.get_ticks()
        if not self.is_ai_thinking:
            self.is_ai_thinking = True
            self.ai_thinking_start = current_time
            self.message = "AI is thinking..."
            return

        # 2. 延迟 1s 后执行
        if current_time - self.ai_thinking_start < 300:
            return

        # 3. 获取动作
        action = self.agent.get_action(self)
        self.is_ai_thinking = False # 思考结束
        
        # 4. 执行动作
        self.perform_ai_action(action)

    def perform_ai_action(self, action_idx):
        ai_p = self.player2
        
        # A. 技能 (54-59)
        if 54 <= action_idx <= 59:
            skill_idx = action_idx - 54
            skill_type_map = {0: SkillType.HEAL, 1: SkillType.STEAL, 2: SkillType.DRAW, 
                              3: SkillType.SHIELD, 4: SkillType.RUIN, 5: SkillType.PIERCE}
            target_type = skill_type_map.get(skill_idx)
            # 找到对应的牌
            for card in ai_p.skill_cards:
                if card.skill_type == target_type:
                    self.use_skill_card(card)
                    return
            # 如果没找到(理论上mask会屏蔽，但以防万一)，不要切换玩家（与 Env_sparse 对齐）
            self.message = "Skill Card not available"
            self.add_action_message("AI tried unavailable skill")
            self.broadcastmessage = "Skill Card not available"
            self.showing_broadcast = True
            self.broadcast_display_time = pygame.time.get_ticks()
            if os.environ.get('AI_DEBUG') == '1':
                print(f"[AI_DEBUG] Skill not available: {target_type}")
            return

        # B. End Turn (60)
        elif action_idx == 60:
            # End Turn: 合法性检查（与 Env_sparse 保持一致）
            if self.player1_round_end or self.player2_round_end:
                # 不合法：不能结束回合
                self.message = "End not legal"
                self.add_action_message("AI tried illegal End Turn")
                self.broadcastmessage = "AI attempted illegal End Turn"
                self.showing_broadcast = True
                self.broadcast_display_time = pygame.time.get_ticks()
                if os.environ.get('AI_DEBUG') == '1':
                    print(f"[AI_DEBUG] Illegal End Turn attempted. player1_round_end={self.player1_round_end}, player2_round_end={self.player2_round_end}")

                # 尝试回退：如果 End Round 合法，执行 End Round
                fallback_mask = self.agent.construct_mask(self)
                if fallback_mask[61] == 1.0:
                    if os.environ.get('AI_DEBUG') == '1':
                        print(f"[AI_DEBUG] Falling back to End Round")
                    self.end_current_round()
                    return

                # 否则，选择其他合法动作（优先 solver 区间），避免重复尝试非法 End Turn
                valid_idxs = [i for i, v in enumerate(fallback_mask) if v == 1.0 and i != 60]
                if valid_idxs:
                    # 优先找 solver 动作 0-53
                    solver_candidates = [i for i in valid_idxs if 0 <= i <= 53]
                    chosen = solver_candidates[0] if solver_candidates else valid_idxs[0]
                    if os.environ.get('AI_DEBUG') == '1':
                        print(f"[AI_DEBUG] Falling back to alternative action {chosen}")
                    # 递归调用 perform_ai_action，但要小心避免深层递归
                    self.perform_ai_action(chosen)
                    return

                # 否则：实在没有可退路，标记 AI 卡住，显示提示并返回
                self.message = "AI stuck: no valid actions"
                self.add_action_message("AI stuck - no valid actions")
                if os.environ.get('AI_DEBUG') == '1':
                    print(f"[AI_DEBUG] AI stuck: fallback_mask={fallback_mask}")
                return

            # 合法：补偿2张并切换玩家
            for _ in range(2): 
                self.current_player.add_card(self.generate_random_card())
            self.switch_player()
            self.broadcastmessage = "AI Ended Turn"
            self.showing_broadcast = True
            self.broadcast_display_time = pygame.time.get_ticks()

        # C. End Round (61)
        elif action_idx == 61:
            self.end_current_round()

        # D. Solver Action (0-53)
        elif 0 <= action_idx <= 53:
            # 1. 解析目标数字
            target_num = None
            t = self.target
            zones = [t.red_zone] + t.yellow_zone[:2] + t.blue_zone[:4]
            spec_map = {0:2, 1:6, 2:120, 3:8, 4:27, 5:64, 
                        6:4, 7:9, 8:16, 9:25, 10:36, 11:49, 12:81, 13:100, 14:121, 15:144, 16:169, 
                        17:24, 18:0, 19:1}
            
            is_std_mode = (action_idx % 2 != 0)
            max_cards = 5 if is_std_mode else 3
            
            if action_idx <= 13:
                z_idx = action_idx // 2
                if z_idx < len(zones): target_num = zones[z_idx]
            else:
                s_idx = (action_idx - 14) // 2
                target_num = spec_map.get(s_idx)
            
            if target_num is None:
                self.switch_player()
                return

            # 2. 调用 Solver 找解
            hand_counts = {}
            for card in ai_p.hand:
                idx = -1
                if card.card_type == CardType.NUMBER: idx = int(card.value)
                elif card.card_type == CardType.OPERATOR: idx = self.agent.str_to_idx.get(card.value)
                if idx != -1: hand_counts[idx] = hand_counts.get(idx, 0) + 1
            
            seq = self.agent.solver.solve(hand_counts, target_num, min_cards=3, max_cards=max_cards)
            
            if seq:
                # 3. 在UI中选中这些牌
                self.selected_cards = []
                temp_hand = list(ai_p.hand)
                
                # 映射回 idx 到 Card 对象
                idx_to_str_op = {14:'+', 15:'-', 16:'*', 17:'/', 18:'(', 19:')'}
                
                for code in seq:
                    if code == 1000: continue # Confirm code
                    
                    found_card = None
                    for i, c in enumerate(temp_hand):
                        c_val = -1
                        if c.card_type == CardType.NUMBER: c_val = int(c.value)
                        elif c.card_type == CardType.OPERATOR: c_val = self.agent.str_to_idx.get(c.value)
                        
                        if c_val == code:
                            found_card = c
                            temp_hand.pop(i) # 避免重复选同一张
                            break
                    
                    if found_card:
                        self.selected_cards.append(found_card)
                
                # 4. 触发 Confirm 逻辑
                # 复制粘贴 handle_click 中的 confirm 逻辑
                if len(self.selected_cards) >= 3:
                    # Confirm 也是一步操作
                    self.step_count += 1
                    
                    result = self.calculate_expression(self.selected_cards)
                    if result is not None:
                        # 人类 confirm 的行为镜像（让 AI 的播报一致）
                        self.step_count += 1
                        self.calculation_result = result
                        dmg = self.target.get_damage(result)
                        actual_dmg = self.player1.take_damage(dmg) # 打玩家1
                        
                        # 构建表达式字符串用于历史和播报
                        expression = "".join([c.value for c in self.selected_cards])
                        skill_trig = self.check_skill_triggers(result)
                        self.add_to_history("AI", expression, result, actual_dmg, skill_trig)

                        # 显示计算结果与播报（与玩家相同）
                        self.calculation_display_time = pygame.time.get_ticks()
                        self.result_alpha = 255
                        self.showing_result = True

                        if actual_dmg > 0:
                            self.play_sound_effect('damage')
                            # 操作消息用于侧边消息栏
                            self.add_action_message(f"AI calculated {result}, causing Player1 {actual_dmg} points of damage!")
                            # 广播消息更详细
                            self.broadcastmessage = f"AI calculated {result} and caused Player1 {actual_dmg} points of damage!"
                            self.broadcast_display_time = pygame.time.get_ticks()
                            self.broadcast_alpha = 255
                            self.showing_broadcast = True
                        elif dmg > 0:
                            self.play_sound_effect('shield_block')

                        # 更新主要信息显示
                        self.message = f"Result: {result}, Damage: {actual_dmg}"
                        if dmg > 0 and actual_dmg == 0:
                            self.message += f" ({dmg} damage blocked by shield)"

                        # 移除牌
                        for c in self.selected_cards:
                            self.current_player.remove_card(c)
                        self.selected_cards = []
                        self.continuous_operations += 1
                    else:
                        # 计算失败：与 Env_sparse 保持一致，不切换玩家，仅提示
                        self.message = "Math Error"
                        self.add_action_message("AI Math Error during calculation")
                        self.broadcastmessage = "AI calculation failed (Math Error)"
                        self.showing_broadcast = True
                        self.broadcast_display_time = pygame.time.get_ticks()
                        if os.environ.get('AI_DEBUG') == '1':
                            print(f"[AI_DEBUG] Math Error with selected_cards={self.selected_cards}")
                        return
            else:
                # 无解：与 Env_sparse 保持一致，不切换玩家，仅提示
                self.message = "Solver Failed (Invalid Action)"
                self.add_action_message("AI Solver Failed (Invalid Action)")
                self.broadcastmessage = "AI solver failed to find a sequence"
                self.showing_broadcast = True
                self.broadcast_display_time = pygame.time.get_ticks()
                if os.environ.get('AI_DEBUG') == '1':
                    print(f"[AI_DEBUG] Solver returned no sequence for target {target_num} with hand_counts={hand_counts}")
                return
    
    def handle_click(self, pos):
        self.play_sound_effect('click')
        """处理鼠标点击"""
        x, y = pos
        # 如果规则窗口显示，只处理关闭按钮
        if self.show_rules:
            # 规则窗口区域
            rules_rect = pygame.Rect(
                SCREEN_WIDTH//2 - 300,
                SCREEN_HEIGHT//2 - 200,
                600,
                400
            )
            
            # 关闭按钮区域
            close_button_rect = pygame.Rect(
                rules_rect.centerx - 50,
                rules_rect.bottom - 60,
                100,
                40
            )
            
            # 检查关闭按钮点击
            if close_button_rect.collidepoint(x, y):
                self.show_rules = False
                return
            
            # 规则窗口显示时不处理其他点击
            return
        
        # 检查规则按钮点击
        if self.buttons['skill'].rect.collidepoint(x, y):
            self.show_rules = True
            return
        

        # 检查当前玩家是否已结束本轮
        if self.current_player == self.player1 and self.player1_round_end:
            self.message = "Player 1 has ended the round, no valid operations"
            return
        if self.current_player == self.player2 and self.player2_round_end:
            self.message = "Player 2 has ended the round, no valid operations"
            return
        
        # 检查手牌点击 根据当前玩家位置
        if self.current_player == self.player1:
            # 玩家1的手牌（左侧）
            for i, card in enumerate(self.current_player.hand):
                card_x, card_y = self.get_card_position(i, 70, SCREEN_HEIGHT - 250)
                if card_x <= x <= card_x + CARD_WIDTH and card_y <= y <= card_y + CARD_HEIGHT:
                    if card in self.selected_cards:
                        self.selected_cards.remove(card)
                    else:
                        self.selected_cards.append(card)
                    return
        else:
            # 玩家2的手牌（右侧）
            for i, card in enumerate(self.current_player.hand):
                card_x, card_y = self.get_card_position(i, 700, SCREEN_HEIGHT - 250)
                if card_x <= x <= card_x + CARD_WIDTH and card_y <= y <= card_y + CARD_HEIGHT:
                    if card in self.selected_cards:
                        self.selected_cards.remove(card)
                    else:
                        self.selected_cards.append(card)
                    return
        
        # 检查技能牌点击 根据当前玩家位置
        if self.current_player == self.player1:
            # 玩家1的技能牌（左侧上方）
            for i, card in enumerate(self.current_player.skill_cards):
                card_x, card_y = self.get_card_position(i, 70, SCREEN_HEIGHT - 300)
                if card_x <= x <= card_x + CARD_WIDTH and card_y <= y <= card_y + CARD_HEIGHT:
                    self.use_skill_card(card)
                    return
        else:
            # 玩家2的技能牌（右侧上方）
            for i, card in enumerate(self.current_player.skill_cards):
                card_x, card_y = self.get_card_position(i, 700, SCREEN_HEIGHT - 300)
                if card_x <= x <= card_x + CARD_WIDTH and card_y <= y <= card_y + CARD_HEIGHT:
                    self.use_skill_card(card)
                    return
        
        # 检查按钮点击（垂直排列）
        if not self.show_rules and self.buttons['confirm'].rect.collidepoint(x, y):  # 确认按钮
            if len(self.selected_cards) >= 3:
                result = self.calculate_expression(self.selected_cards)
                if result is not None and result >= 0:
                    # 人类 confirm 也是一步
                    self.step_count += 1
                    self.calculation_result = result
                    damage = self.target.get_damage(result)
                    other_player = self.player2 if self.current_player == self.player1 else self.player1
                    actual_damage = other_player.take_damage(damage)
                    
                    # 构建表达式字符串
                    expression = "".join([card.value for card in self.selected_cards])
                    
                    # 检查技能触发
                    skill_triggered = self.check_skill_triggers(result)
                    
                    # 添加到历史记录
                    self.add_to_history(
                        self.current_player.name, 
                        expression, 
                        result, 
                        actual_damage, 
                        skill_triggered
                    )
                    
                    self.calculation_display_time = pygame.time.get_ticks()  # 记录开始显示时间
                    self.result_alpha = 255  # 重置为完全不透明
                    self.showing_result = True
                    #self.broadcastmessage = ""

                    # 添加操作消息
                    if actual_damage > 0:
                        self.play_sound_effect('damage')
                        self.add_action_message(f"{self.current_player.name} calculated {result}, causing {other_player.name} {actual_damage} points of damage!")
                        self.broadcastmessage = f"{self.current_player.name} calculated {result} and caused {other_player.name} {actual_damage} points of damage!"
                    
                     # 添加广播消息
                    self.broadcast_display_time = pygame.time.get_ticks()
                    self.broadcast_alpha = 255
                    self.showing_broadcast = True
                            
                    
                    self.message = f"Result: {result}, Damage: {actual_damage}"
                    if damage > 0 and actual_damage == 0:
                        self.message += f" ({damage} damage blocked by shield)"
                        self.play_sound_effect('shield_block')
                    
                    # 移除使用的卡牌
                    for card in self.selected_cards:
                        self.current_player.remove_card(card)
                    
                    self.selected_cards = []
                    self.continuous_operations += 1
                    #self.message += f" (Operate continuously for {self.continuous_operations} times)"
                else:
                    self.message = "Invalid expression"
            else:
                self.message = "Need at least 3 cards"
        
        if not self.show_rules and self.buttons['end'].rect.collidepoint(x, y):  # 结束按钮 结束连续操作
            if not self.player1_round_end and not self.player2_round_end:
                makeuplist = []
                for _ in range(2):
                    makeupcard = self.generate_random_number_card()
                    self.current_player.add_card(makeupcard)
                    makeuplist.append(makeupcard.value)
                for _ in range(1):
                    makeupcard = self.generate_random_op_card()
                    self.current_player.add_card(makeupcard)
                    makeuplist.append(makeupcard.value)
                self.broadcastmessage = f"{self.current_player.name} ended continuous operations and drew {' '.join(makeuplist)}"
                self.broadcast_display_time = pygame.time.get_ticks()
                self.broadcast_alpha = 255
                self.showing_broadcast = True
            self.switch_player()
        
        if not self.show_rules and self.buttons['end_round'].rect.collidepoint(x, y): # 结束本轮按钮
            self.end_current_round()
        
    def draw_card(self, card: Card, x: int, y: int, selected: bool = False, alpha: int = 255):
        #绘制卡牌
        color = WHITE if not selected else YELLOW
        if alpha < 255:
            # 半透明表面
            card_surface = pygame.Surface((CARD_WIDTH, CARD_HEIGHT))
            card_surface.set_alpha(alpha)
            card_surface.fill(color)
            self.screen.blit(card_surface, (x, y))
        else:
            pygame.draw.rect(self.screen, color, (x, y, CARD_WIDTH, CARD_HEIGHT))
        
        pygame.draw.rect(self.screen, BLACK, (x, y, CARD_WIDTH, CARD_HEIGHT), 1)
        
        text = self.card_font.render(card.value, True, BLACK)
        text_rect = text.get_rect(center=(x + CARD_WIDTH//2, y + CARD_HEIGHT//2))
        self.screen.blit(text, text_rect)

    def update(self):
        # 更新结果显示状态
        if self.showing_result:
            current_time = pygame.time.get_ticks()
            elapsed = current_time - self.calculation_display_time
            
            if elapsed > 2600:  # 2.4秒后开始淡出
                fade_progress = min(1.0, (elapsed - 2600) / self.fade_duration)
                # 使用缓动函数使动画更自然
                fade_progress = math.sin(fade_progress * math.pi/2)  
                self.result_alpha = int(255 * (1 - fade_progress))
                
                if fade_progress >= 1.0:
                    self.showing_result = False
            # 更新广播消息显示状态
        if self.showing_broadcast or self.showing_result:
            current_time = pygame.time.get_ticks()
            elapsed = current_time - self.broadcast_display_time
        
            if elapsed > 2400:  # 2.4秒后开始淡出
                fade_progress = min(1.0, (elapsed - 2400) / self.fade_duration)
                fade_progress = math.sin(fade_progress * math.pi/2)  
                self.broadcast_alpha = int(255 * (1 - fade_progress))
                
                if fade_progress >= 1.0:
                    self.showing_broadcast = False
                    self.broadcastmessage = ""  # 清空消息
        
    def draw(self):
        """绘制游戏界面"""
        # 在加载背景图片后添加调试信息
        background = pygame.image.load("BAckground.png").convert()
        # 直接拉伸到屏幕尺寸（图片会变形）
        background = pygame.transform.scale(background, (SCREEN_WIDTH, SCREEN_HEIGHT))
        self.screen.blit(background, (0, 0))

        # 绘制标题
        title_font = pygame.font.SysFont('Arial', 60, bold=True)
        title = title_font.render("CALC WARS", True, (0, 0, 128))  # 深蓝色标题
        self.screen.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, 10))

        
        # 绘制玩家信息
        player1_text = f"Player 1 HP: {self.player1.hp} Shield: {self.player1.shield_count}"
        player2_text = f"Player 2 HP: {self.player2.hp} Shield: {self.player2.shield_count}"
        
    
        self.screen.blit(self.font.render(player1_text, True, GOLD_CLASSIC), (65, SCREEN_HEIGHT - 340))
        self.screen.blit(self.font.render(player2_text, True, GOLD_CLASSIC), (695, SCREEN_HEIGHT - 340))

        #绘制血槽
        health_bar_width = 200
        health_bar_height = 20
        current_health_width1 = (self.player1.hp / 120) * health_bar_width
        pygame.draw.rect(self.screen, YELLOW_CREAM, (65, SCREEN_HEIGHT - 360, health_bar_width, health_bar_height))
        pygame.draw.rect(self.screen, RED_ELECTRIC, (65, SCREEN_HEIGHT - 360, current_health_width1, health_bar_height))
        #绘制边框
        pygame.draw.rect(self.screen, BLACK, (65, SCREEN_HEIGHT - 360, health_bar_width, health_bar_height), 2)

        current_health_width2 = (self.player2.hp / 120) * health_bar_width
        pygame.draw.rect(self.screen, YELLOW_CREAM, (695, SCREEN_HEIGHT - 360, health_bar_width, health_bar_height))
        pygame.draw.rect(self.screen, RED_ELECTRIC, (695, SCREEN_HEIGHT - 360, current_health_width2, health_bar_height))
        #绘制边框
        pygame.draw.rect(self.screen, BLACK, (695, SCREEN_HEIGHT - 360, health_bar_width, health_bar_height), 2)
        
        #绘制盾量
        shield_img = pygame.image.load("Shield.png").convert_alpha()
        # 调整大小（如果需要）
        shield_img = pygame.transform.scale(shield_img, (25, 25))
        for i in range(self.player1.shield_count):
            self.screen.blit(shield_img, (265 + i * 27, SCREEN_HEIGHT - 365))
        for i in range(self.player2.shield_count):
            self.screen.blit(shield_img, (895 + i * 27, SCREEN_HEIGHT - 365))

        # 绘制目标区域
        def render_target_text(screen, font, x, y, target):
            color_map = {
                "Red": (255, 50, 50),
                "Yellow": (255, 255, 50),
                "Blue": (180, 240, 255)
            }
            
            parts = [
                ("Red: ", color_map["Red"]),
                (str(target.red_zone), color_map["Red"]),
                ("   Yellow: ", color_map["Yellow"]),
                (str(target.yellow_zone), color_map["Yellow"]),
                ("   Blue: ", color_map["Blue"]),
                (str(target.blue_zone), color_map["Blue"])
            ]
            
            current_x = x
            for text, color in parts:
                text_surface = font.render(text, True, color)
                screen.blit(text_surface, (current_x, y))
                current_x += text_surface.get_width()+ 10
        render_target_text(self.screen, self.large_font, 230, 185, self.target)
        
        # 绘制当前玩家和游戏状态
        current_text = f"Current Player: {self.current_player.name}"
        self.screen.blit(self.font.render(current_text, True, WHITE_IVORY), (700, 80))
        
        # 绘制连续操作次数
        if self.continuous_operations > 0:
            ops_text = f"Operate continuously for {self.continuous_operations} times"
            self.screen.blit(self.small_font.render(ops_text, True, WHITE_IVORY), (700, 150))
        
        # 绘制本轮结束状态
        round_status = f"Player 1 ends the round: {'YES' if self.player1_round_end else 'NO'} | Player 2 ends the round: {'YES' if self.player2_round_end else 'NO'}"
        self.screen.blit(self.small_font.render(round_status, True, WHITE_CREAM), (700, 110))

        #绘制轮次
        self.newcardsnum = 5 + int((self.round_number+ 1) / 2)
        round_text = f"Round: {self.round_number}  Next Round New Cards Each: {self.newcardsnum}"
        self.screen.blit(self.small_font.render(round_text, True, WHITE_IVORY), (700, 130))
        
        # 绘制消息
        self.screen.blit(self.large_font.render(self.message, True, YELLOW_SUN), (280, 250))
        
        # 绘制按钮（垂直排列）
        for button in self.buttons.values():
            button.draw(self.screen)
        
        # 绘制双方手牌（分得更开）
        # 玩家1的手牌（左侧）
        for i, card in enumerate(self.player1.hand):
            x, y = self.get_card_position(i, 70, SCREEN_HEIGHT - 250)
            if self.current_player == self.player1:
                # 当前玩家，可操作
                selected = card in self.selected_cards
                self.draw_card(card, x, y, selected)
            else:
                # 非当前玩家，半透明
                self.draw_card(card, x, y, False, 128)
        
        # 玩家2的手牌（右侧）
        for i, card in enumerate(self.player2.hand):
            x, y = self.get_card_position(i, 700, SCREEN_HEIGHT - 250)
            if self.current_player == self.player2:
                # 当前玩家，可操作
                selected = card in self.selected_cards
                self.draw_card(card, x, y, selected)
            else:
                # 非当前玩家，半透明
                self.draw_card(card, x, y, False, 128)
        
        # 绘制技能牌（分别显示在对应玩家上方）
        # 玩家1的技能牌（左侧上方）
        for i, card in enumerate(self.player1.skill_cards):
            x, y = self.get_card_position(i, 70, SCREEN_HEIGHT - 300)
            if self.current_player == self.player1:
                # 当前玩家，可操作
                self.draw_card(card, x, y)
            else:
                # 非当前玩家，半透明
                self.draw_card(card, x, y, False, 128)
        
        # 玩家2的技能牌（右侧上方）
        for i, card in enumerate(self.player2.skill_cards):
            x, y = self.get_card_position(i, 700, SCREEN_HEIGHT - 300)
            if self.current_player == self.player2:
                # 当前玩家，可操作
                self.draw_card(card, x, y)
            else:
                # 非当前玩家，半透明
                self.draw_card(card, x, y, False, 128)
        
        # 绘制历史记录
        history_y = 50
        history_x = 50
        self.screen.blit(self.font.render("History Message:", True, GOLD_LIGHT), (history_x, history_y))
        for i, entry in enumerate(self.history[-5:]):  # 显示最近5条记录
            history_text = f"{entry['player']}: {entry['expression']}={entry['result']} (Damage:{entry['damage']})"
            if entry['skill']:
                history_text += f" [{entry['skill']}]"
            self.screen.blit(self.small_font.render(history_text, True, BLACK), (history_x, history_y + 25 + i * 20))
        
        # 绘制操作消息
        messages_y = history_y + 128
    
        # 绘制选中的卡牌
        if self.selected_cards:
            selected_text = " ".join(card.value for card in self.selected_cards)
            self.screen.blit(self.super_large_font.render(selected_text, True, WHITE_PURE), (350, messages_y + 200))

        #绘制计算结果与操作消息
        if self.showing_broadcast or self.showing_result:
            Operation = self.broadcastmessage if self.broadcastmessage else ""

            operation_surface = self.large_font.render(Operation, True, YELLOW_BRIGHT)
            alpha_surface_op = pygame.Surface(operation_surface.get_size(), pygame.SRCALPHA)
            alpha_surface_op.blit(operation_surface, (0, 0))
            alpha_surface_op.set_alpha(self.broadcast_alpha)

            self.screen.blit(alpha_surface_op, (220, messages_y + 250))

        if self.showing_result:
            # 计算已经显示的时间
            result_text = f"Calculation Result:  {self.calculation_result}"
            # 创建带透明度的表面
            text_surface = self.super_large_font.render(result_text, True, ORANGE_SAFETY)
            alpha_surface = pygame.Surface(text_surface.get_size(), pygame.SRCALPHA)
            alpha_surface.blit(text_surface, (0, 0))
            alpha_surface.set_alpha(self.result_alpha)
            
            self.screen.blit(alpha_surface, (370, messages_y + 140))

        # 绘制技能牌获取方法说明
        
        
        if self.show_rules:
            # 获取鼠标位置
            mouse_pos = pygame.mouse.get_pos()

            # 半透明背景
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            # 规则窗口
            rules_rect = pygame.Rect(
                SCREEN_WIDTH//2 - 300,
                SCREEN_HEIGHT//2 - 200,
                600,
                400
            )
            pygame.draw.rect(self.screen, (255, 255, 255), rules_rect)
            pygame.draw.rect(self.screen, (0, 0, 0), rules_rect, 3)
            
            # 规则标题
            title = self.font.render("Skill Cards", True, (0, 0, 0))
            self.screen.blit(title, (rules_rect.centerx - title.get_width()//2, rules_rect.y + 20))
            
            # 规则内容
            for i, line in enumerate(self.rules_text):
                text = self.small_font.render(line, True, (0, 0, 0))
                self.screen.blit(text, (rules_rect.x + 30, rules_rect.y + 70 + i * 30))
            
            # 关闭按钮
            close_button = Button(
                rules_rect.centerx - 50,
                rules_rect.bottom - 60,
                100,
                40,
                "Close",
                (0, 0, 255),
                (100, 100, 255)
            )

            # 更新关闭按钮悬停状态
            close_button.check_hover(mouse_pos)
        
            close_button.draw(self.screen)

        pygame.display.flip()

    
    def run(self):
        """运行游戏"""
        #self.play_background_music()  # 开始播放背景音乐
        # 开始播放第一首音乐
        self.play_next_music()
        running = True
    
        while running:
            mouse_pos = pygame.mouse.get_pos()
            mouse_pressed = pygame.mouse.get_pressed()  # 获取当前鼠标状态
        
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  
                        self.handle_click(event.pos)

            # 更新所有按钮状态（每次循环都更新）
            for button in self.buttons.values():
                button.check_hover(mouse_pos)
                # 更新按下状态（如果鼠标左键按下且悬停在按钮上）
                button.is_pressed = mouse_pressed[0] and button.is_hovered

            # 检查当前音乐是否播放完毕，如果是则播放下一首
            if not pygame.mixer.music.get_busy():
                self.play_next_music()

            if self.current_player == self.player2:
            # 只有当不在显示结果动画时才行动，避免视觉混乱
                self.execute_ai_turn()
        
            self.update()
            self.draw()
            self.clock.tick(60)

            # 检查游戏结束
            if self.player1.hp <= 0:
                self.message = "Player 2 Wins!"
                win_font = pygame.font.SysFont('Arial', 240, bold=True)
                win = win_font.render("Player2 WINS!", True, ORANGE_PEACH)  # 深蓝色标题
                self.play_sound_effect('win')
                self.screen.blit(win, (0, 300))
                pygame.display.flip()  # 确保消息显示
                pygame.time.wait(5000)
                running = False
            elif self.player2.hp <= 0:
                self.message = "Player 1 Wins"
                win_font = pygame.font.SysFont('Arial', 240, bold=True)
                win = win_font.render("Player1 WINS!", True, ORANGE_PEACH)  # 深蓝色标题
                self.play_sound_effect('win')
                self.screen.blit(win, (0, 300))
                pygame.display.flip()  # 确保消息显示
                pygame.time.wait(5000)
                running = False
        

if __name__ == "__main__":
    game = Game()
    game.run()
