"""
AI_opponents.Env_sparse 的 Docstring
环境定义：稀疏奖励版本的计算战争游戏
对于主游戏完全去除UI并且numpy向量化
"""
import pygame
import random
import math
from enum import Enum
from typing import List, Dict, Tuple, Optional
import os
import numpy as np
import itertools
from Smart_solver import FastTemplateSolver

class CardType(Enum):
    NUMBER = "number"
    OPERATOR = "operator"
    SKILL = "skill"

class OperatorType(Enum):
    PLUS = 14          # 加法
    MINUS = 15         # 减法
    MULTIPLY = 16      # 乘法
    DIVIDE = 17        # 除法
    LEFTBRA = 18   #左括号
    RIGHTBRA = 19  #右括号

class SkillType(Enum):
    HEAL = 0          # 生命恢复牌
    STEAL = 1      # 盗窃牌
    DRAW = 2          # 抽牌
    SHIELD = 3      # 护盾牌
    RUIN = 4   #0牌
    PIERCE = 5 #1牌

class Card:
    def __init__(self, card_type: CardType, 
                 value: Optional[int] = None,
                 skill_type: Optional[SkillType] = None,
                 operator_type: Optional[OperatorType] = None):
        self.card_type = card_type
        self.value = value  # 对于数字卡，这是数字值
        self.skill_type = skill_type
        self.operator_type = operator_type
        self.used = False

    def __eq__(self, other):
        if not isinstance(other, Card):
            return False
        return (self.card_type == other.card_type and 
                self.value == other.value and
                self.skill_type == other.skill_type and
                self.operator_type == other.operator_type)
    
    def __hash__(self):
        return hash((self.card_type, self.value, self.skill_type, self.operator_type))

    def get_hand_index(self) -> int:
        """返回卡牌在列表中的索引位置"""
        if self.card_type == CardType.NUMBER:
            return self.value  # 数字卡直接使用其值作为索引(0-13)
        elif self.card_type == CardType.OPERATOR:
            if self.operator_type is None:
                raise ValueError("Operator type is None for operator card")
            return self.operator_type.value  # 运算符卡放在数字卡之后(14-19)
        else:
            raise ValueError("NOT FOUND")
        
    def get_skill_index(self) -> int:
        """返回卡牌在技能牌列表中的索引位置"""
        if self.card_type == CardType.SKILL:
            return self.skill_type.value  # 技能卡使用预定义的索引(0-5)
        else:
            raise ValueError("NOT FOUND")
    
    def __str__(self):
        if self.card_type == CardType.NUMBER:
            return str(self.value)
        elif self.card_type == CardType.OPERATOR:
            if self.operator_type is None:
                return "INVALID_OPERATOR"
            return self.operator_type.name
        else:  # SKILL
            if self.skill_type is None:
                return "INVALID_SKILL"
            return self.skill_type.name

class Player:
    def __init__(self, name: str):
        self.name = name
        self.hp = 120
        self.hand = [0] * 20    # 手牌（数字和运算符）
        self.skill_cards = [0] * 6  # 技能卡
        self.shield_count = 0
        self.is_active = False

        
    def add_card(self, card: Card):
        """添加卡牌到对应牌库"""
        if card.card_type == CardType.SKILL:
            index = card.get_skill_index()
            if 0 <= index < len(self.skill_cards):
                self.skill_cards[index] += 1
            else:
                print(f"Warning: Invalid skill card index {index}")
        else:  # 数字卡或运算符卡
            index = card.get_hand_index()
            if 0 <= index < len(self.hand):
                self.hand[index] += 1
            else:
                print(f"Warning: Invalid hand card index {index}")

    def remove_card(self, card: Card):
        """从牌库中移除一张卡牌"""
        if card.card_type == CardType.SKILL:
            index = card.get_skill_index()
            if 0 <= index < len(self.skill_cards) and self.skill_cards[index] > 0:
                self.skill_cards[index] -= 1
        else:  # 数字卡或运算符卡
            index = card.get_hand_index()
            if 0 <= index < len(self.hand) and self.hand[index] > 0:
                self.hand[index] -= 1
    
    def has_card(self, card: Card) -> bool:
        """检查是否拥有某张卡牌"""
        if card.card_type == CardType.SKILL:
            index = card.get_skill_index()
            return 0 <= index < len(self.skill_cards) and self.skill_cards[index] > 0
        else:
            index = card.get_hand_index()
            return 0 <= index < len(self.hand) and self.hand[index] > 0
    
    def get_card_count(self, card: Card) -> int:
        """获取某张卡牌的数量"""
        if card.card_type == CardType.SKILL:
            index = card.get_skill_index()
            if 0 <= index < len(self.skill_cards):
                return self.skill_cards[index]
            return 0
        else:
            index = card.get_hand_index()
            if 0 <= index < len(self.hand):
                return self.hand[index]
            return 0
    
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
        self.yellow_zone = random.sample([24,26,27,28,29,30,31,32,33,34,35],2)
        self.blue_zone = random.sample([1,2,3,5,6,7,8,10,11,12,13,14,15,17,18,19,20,21,22,23],3)+ random.sample([4,9,16],1)
    
    def get_damage(self, result: int) -> int:
        if result == self.red_zone:
            return 50
        elif result in self.yellow_zone:
            return 30
        elif result in self.blue_zone:
            return 10
        return 0
    
    def set_zones(self, zones):
        self.red_zone = zones[0]
        self.yellow_zone = zones[1:3]
        self.blue_zone = zones[3:7]
    

class Game:
    def __init__(self):
        self.seq_len = 9
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
        self.done = False

        self.step_count = 0
        self.state_vector = np.zeros(128, dtype=np.float32)
        self.info = {"message": ""}

        # 统计指标 (用于 WandB)
        self.stats = {
            'solver_attempts': 0,   # 尝试调用 Solver 次数 (Action 0-53) 只统计主模型
            'solver_success': 0,    # Solver 成功找到解的次数
            'expr_length': 0,       # 表达式长度
            'damage_dealt': 0,      # 总造成伤害
            'skill_triggered': 0,   # 技能触发次数 (Combo)
            'skill_used': 0,        # 主动技能使用次数
            'rounds': 0,             # 轮次数量
            'turns' : 0             # 回合数量
        }

        self.nonaction_times = 0
        self.noneffect_times = 0

        self.symbolic_executor = SymbolicExecutor()
        self.solver = FastTemplateSolver()
        self.trainer = None  # 将由TrainingManager注入
        # 初始化游戏
        self.initial_deal()


    def reset(self):
        self.player1.hand = [0] * 20
        self.player2.hand = [0] * 20
        self.player1.skill_cards = [0] * 6
        self.player2.skill_cards = [0] * 6

        self.player1.hp = 120
        self.player2.hp = 120
        
        # 同时重置护盾
        self.player1.shield_count = 0
        self.player2.shield_count = 0

        self.target = Target()
        self.game_phase = "initial_deal"  # initial_deal, playing, round_end
        self.round_number = 1
        self.turn_number = 1
        self.selected_cards = []
        self.calculation_result = None
        self.done = False
        self.player1_round_end = False  # 玩家1是否结束本轮
        self.player2_round_end = False  # 玩家2是否结束本轮
        self.continuous_operations = 0  # 连续操作次数
        self.first_to_end_round = None  # 当前轮次先结束本轮的玩家

        self.step_count = 0
        self.state_vector = np.zeros(128, dtype=np.float32)
        self.leftright_unbal_bra = 0

        self.nonaction_times = 0
        self.noneffect_times = 0

        self.symbolic_executor.reset()
        # 清空统计
        self.stats.update((key, 0) for key in self.stats) 

        # 重新发牌
        self.initial_deal()

        return self._get_state()


    def _get_state(self):
        """获取当前游戏状态的数值表示 - 优化版本"""
        # 预分配整个状态向量
        self.state_vector = np.zeros(92, dtype=np.float32)

        opponent = self.player2 if self.current_player == self.player1 else self.player1
        
        # 手牌信息
        # 当前玩家手牌 
        self.state_vector[:20] = np.array(self.current_player.hand[:20], dtype=np.float32) / 5.0
        
        # 对手手牌 
        self.state_vector[20:40] = np.array(opponent.hand[:20], dtype=np.float32) / 5.0
        
        # 技能牌信息
        # 当前玩家技能 
        self.state_vector[40:46] = np.array(self.current_player.skill_cards[:6], dtype=np.float32) / 3.0
        
        # 对手技能 
        self.state_vector[46:52] = np.array(opponent.skill_cards[:6], dtype=np.float32) / 3.0
        
        # 核心信息
        self.state_vector[52] = self.current_player.hp / 120.0
        self.state_vector[53] = opponent.hp / 120.0
        self.state_vector[54] = self.current_player.shield_count / 4.0
        self.state_vector[55] = opponent.shield_count / 4.0
        self.state_vector[56] = self.round_number / 5.0
        self.state_vector[57] = self.turn_number / 10.0
        self.state_vector[58] = self.step_count / 100.0
        self.state_vector[59] = self.continuous_operations / 5.0
        self.state_vector[60] = (self.state_vector[:20].sum() - self.state_vector[20:40].sum())/10.0 #手牌差距
        self.state_vector[61] = (self.current_player.hp - opponent.hp) / 50.0 #血量差距
        if self.first_to_end_round == self.current_player:
            self.state_vector[62] = 1.0  # 我是先手结束方
        elif self.first_to_end_round == opponent:
            self.state_vector[62] = -1.0 # 我是后手结束方
        else:
            self.state_vector[62] = 0.0
        
        if self.current_player == self.player1:
            self.state_vector[63] = 1.0 if self.player1_round_end else 0.0
            self.state_vector[64] = 1.0 if self.player2_round_end else 0.0
        else:
            self.state_vector[63] = 1.0 if self.player2_round_end else 0.0
            self.state_vector[64] = 1.0 if self.player1_round_end else 0.0
    
        #  目标区域信息
        zones = [self.target.red_zone] +  self.target.yellow_zone[:2] +  self.target.blue_zone[:4] 
        self.state_vector[65:72] = np.array(zones[:7], dtype=np.float32) / 40.0 #归一化修改，放大注意

        # 高级特征提取
        extract_feature = np.zeros(18, dtype=np.float32)
        
        # 资源结构分析
        nums = []
        ops = []
        for i in range(14): nums.extend([i] * self.current_player.hand[i])
        for i in range(14, 18): ops.extend([i] * self.current_player.hand[i]) # 不含括号
        has_bracket = (self.current_player.hand[18] > 0 and self.current_player.hand[19] > 0)
        
        num_cnt = len(nums)
        op_cnt = len(ops)
        
        #  数字数量 (归一化)
        extract_feature[0] = num_cnt / 10.0
        #  符号数量 (归一化)
        extract_feature[1] = op_cnt / 5.0
        #  卡手指数 (Op Density). 理想值是 0.5-0.8 左右 (1个符号配2个数字)
        # 如果 > 1.0 说明符号太多，必须泄洪
        extract_feature[2] = op_cnt / (num_cnt + 0.1)

        # 关键目标匹配 (Direct Match)
        # 得益于我们左括号数字右括号的特殊规则
        #  蓝色区域直接匹配
        extract_feature[3:7] = [1.0 if zones[i+3] in nums else 0.0 for i in range(4)]
        #  技能数字直接匹配 
        skill_targets = [0, 1, 2, 4, 6, 8, 9]
        extract_feature[7:14] = [1.0 if skill_targets[i] in nums else 0.0 for i in range(7)]
        extract_feature[14] = 1.0 if has_bracket else 0.0

        # 技能Hint
        # 偷牌收益如何
        extract_feature[15] = 1.0 if self.state_vector[:20].sum() - self.state_vector[20:40].sum() <= 6.0 else 0.0
        # 是否需要Pierce
        extract_feature[16] = 1.0 if opponent.shield_count >= 2.0 else 0.0
        
        #  斩杀提示
        extract_feature[17] = 1.0 if opponent.hp <= 50.0 else 0.0

        self.state_vector[72:90] = extract_feature[:18]
 
        # 游戏结束标志
        self.state_vector[91] = 1.0 if self.done else 0.0
        
        return self.state_vector
    
    
    def get_oracle_mask(self):
        """
        优化版 Mask 计算：使用 Forward Pass 极速生成
        """
        full_mask = np.zeros(62, dtype=np.float32)

        skills = np.array(self.current_player.skill_cards[:6])
        
        # 1. Skill & Flow (不变)
        full_mask[54:60] = (skills > 0).astype(np.int8)
        full_mask[60] = 1.0 
        if self.player1_round_end or self.player2_round_end:
            full_mask[60] = 0.0
        full_mask[61] = 1.0
        
        # 2. Solver 动作优化
        hand_counts = {i: c for i, c in enumerate(self.current_player.hand) if c > 0}
        total_cards = sum(hand_counts.values())
        
        if total_cards < 3:
            full_mask[61] = 1.0
            return full_mask 
            
        # --- 核心优化: 一次性生成所有解 ---
        # reachable_3: 3张牌能凑出的数字集合
        # reachable_5: 5张牌能凑出的数字集合
        reachable_3, reachable_5 = self.solver.get_reachable_sets(hand_counts)
        
        # 目标映射表
        zones = [self.target.red_zone] + self.target.yellow_zone + self.target.blue_zone
        spec_map = {0:2, 1:6, 2:120, 3:8, 4:27, 5:64, 
                    6:4, 7:9, 8:16, 9:25, 10:36, 11:49, 12:81, 13:100, 14:121, 15:144, 16:169, 
                    17:24, 18:0, 19:1}

        for i in range(27):
            target = None
            if i < 7: target = zones[i]
            else: target = spec_map.get(i - 7)
            
            if target is None: continue
            
            # 直接查表，O(1) 复杂度
            # 注意: target 是 float/int，集合里为了避开浮点误差存的是 round 后的值
            # 建议在查找时也 round 一下，或者在 solver 里存的时候转 int (如果都是整数)
            # 这里的 target 都是整数，可以直接查
            
            is_eco = target in reachable_3
            
            # Std (3-5张) = 3张有解 OR 5张有解
            is_std = is_eco or (target in reachable_5)
            
            base_idx = 2 * i if i < 7 else 14 + 2 * (i - 7)
            if is_eco: full_mask[base_idx] = 1.0
            if is_std: full_mask[base_idx + 1] = 1.0
        
        if full_mask.sum() == 0:
            full_mask[61] = 1.0
            
        return full_mask
    
    
    def step(self, raw_action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        原子化 Step: 集成 Translator + Executor
        raw_action: 0-61
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
        opponent = self.player2 if self.current_player == self.player1 else self.player1
        reward = 0.0
        self.step_count += 1
        time_penalty = -0.00008
        reward += time_penalty

        # 检查游戏是否已经结束
        if self.done:
            return self._get_state(), 0, True, {"message": "Game already finished"}

        # === 1. 技能卡动作 (54-59) ===
        if 54 <= raw_action <= 59:
            skill_index = raw_action - 54 # 映射回 0-5
            if self.current_player.skill_cards[skill_index] > 0:
                skill_type = SkillType(skill_index)
                skill_card = Card(CardType.SKILL, skill_type=skill_type)
                
                self.info["message"] = f"Used skill: {skill_type.name}"
                if self.current_player == self.player1:
                    self.stats['skill_used'] += 1

                if raw_action == 58 or raw_action == 55: # RUIN STEAL 不准对手没牌
                    if sum(opponent.hand[:20]) <= 1:
                        reward -= 0.1
                        self.info["message"] = "Skill Card not effective"

                
                self.use_skill_card(skill_card)
            else:
                reward = -0.01
                self.info["message"] = "Skill Card not available"

        # === 2. 结束回合/轮次 (60-61) ===
        elif raw_action == 60: # End Turn
            if self.continuous_operations <= 1:
                reward -= 0.0005
            if sum(self.current_player.hand[:20]) - sum(opponent.hand[:20]) <= -4:
                reward -= 0.002
            elif sum(self.current_player.hand[:20]) - sum(opponent.hand[:20]) >= 4:
                reward += 0.0
            if not self.player1_round_end and not self.player2_round_end:
                self.turn_number += 1

                for _ in range(2):  # 结束回合补2张牌
                    makeup_card = self.generate_random_card()
                    self.current_player.add_card(makeup_card)
                self.switch_player()
                self.info["message"] = "Ended turn" 
            else:
                self.info["message"] = "End not legal"
                reward += -0.01
                self.nonaction_times += 1

        elif raw_action == 61: # End Round
            if sum(self.current_player.hand[:20]) - sum(opponent.hand[:20]) <= -4:
                reward += 0.002
            self.turn_number += 1
            self.end_current_round()
            self.info["message"] = "Ended round"

        # === 3. Solver 动作 (0-53) - 原子化执行核心 ===
        elif 0 <= raw_action <= 53:
            if self.current_player == self.player1:
                self.stats['solver_attempts'] += 1
            
            # --- A. 内部解析 Target ---
            is_std_mode = (raw_action % 2 != 0)
            max_cards = 5 if is_std_mode else 3
            min_cards = 3
            
            zones = [self.target.red_zone] + self.target.yellow_zone[:2] + self.target.blue_zone[:4]
            target_num = None
            if raw_action <= 13: # 区域数字
                zone_index = int((raw_action + 1e-3) // 2)
                if zone_index < len(zones):
                    target_num = zones[zone_index]

            else: # 特殊数字
                special_num_op = {0:2, 1:6, 2:120, 3:8, 4:27, 5:64, 
                                  6:4, 7:9, 8:16, 9:25, 10:36, 11:49, 12:81, 13:100, 14:121, 15:144, 16:169, 
                                  17:24, 18:0, 19:1}
                special_index = int((raw_action - 14.0 + 1e-3) // 2)
                target_num = special_num_op.get(special_index)

            # --- B. 内部调用 Solver (Atomic) ---
            select_sequence = None
            if target_num is not None:
                # 直接使用当前的 self.current_player.hand，绝无不同步可能
                hand_counts = dict(enumerate(self.current_player.hand[:20]))
                select_sequence = self.solver.solve(
                    hand_counts=hand_counts,
                    target=target_num,
                    min_cards=min_cards,
                    max_cards=max_cards
                )

            # --- C. 执行验证 ---
            if not select_sequence:
                reward = -0.01
                self.info["message"] = "Solver Failed (Invalid Action)"
            else:
                # 序列转对象
                selected_cards = []
                temp_hand = list(self.current_player.hand)
                valid_seq = True
                
                for idx in select_sequence:
                    if idx == 1000: continue
                    if idx >= 20: continue
                    
                    if temp_hand[idx] > 0:
                        temp_hand[idx] -= 1
                        if idx <= 13: 
                            selected_cards.append(Card(CardType.NUMBER, value=idx))
                        else: 
                            selected_cards.append(Card(CardType.OPERATOR, operator_type=OperatorType(idx)))
                    else:
                        valid_seq = False
                        break
                
                if not valid_seq or len(selected_cards) < 3:
                    reward = -0.01
                    self.info["message"] = "Invalid Sequence Construction"
                else:
                    # 计算结果
                    result = self.calculate_expression(selected_cards)
                    
                    if result is not None and result >= 0: # 前面已经加了 calculate_expression 的容错
                        # === 执行成功 ===
                        if self.current_player == self.player1:
                            self.stats['solver_success'] += 1
                            self.stats['expr_length'] += len(select_sequence) - 1
                        
                        # 真正的扣牌
                        for idx in select_sequence:
                            if idx != 1000 and idx < 20:
                                self.current_player.hand[idx] -= 1
                        
                        damage = self.target.get_damage(result)
                        actual_damage = opponent.take_damage(damage)
                        reward += actual_damage * 0.0 # 伤害奖励
                        if self.current_player == self.player1:
                            self.stats['damage_dealt'] += actual_damage


                        skill_messag = self.check_skill_triggers(result)
                        if skill_messag: 
                            self.info['skill'] = skill_messag
                            if self.current_player == self.player1:
                                self.stats['skill_triggered'] += 1

                        self.info['message'] = f"Damage {actual_damage}"
                        self.continuous_operations += 1
                    else:
                        reward = -0.01
                        self.info['message'] = "Math Error"
                        # 原子操作下，这里报错说明 calculate_expression 和 solver._calc_vec 还是有精度偏差
                        # 但之前 V4 应该已经解决了

        self.stats['rounds'] = self.round_number
        self.stats['turns'] = self.turn_number

        if self.player1.hp <= 0 or self.player2.hp <= 0:
            self.done = True
            if self.current_player.hp > 0:
                reward = 1.0
            else:
                reward = -1.0
        
        return self._get_state(), reward, self.done, self.info
    

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
        if self.current_player == self.player2:
            ruin_card = self.generate_skill_card(SkillType.RUIN)
            self.player1.add_card(ruin_card)
        else:
            ruin_card = self.generate_skill_card(SkillType.RUIN)
            self.player2.add_card(ruin_card)
        
        self.current_player.is_active = True
        self.game_phase = "playing"
        self.message = f"{self.current_player.name} Starts the Round"
        self.player1_round_end = False
        self.player2_round_end = False
        self.continuous_operations = 0
    
    def generate_random_card(self) -> Card:
        """生成随机卡牌"""
        if random.random() < 0.66667:  # 66.667%概率数字牌
            number_value = random.randint(0, 13)
            return Card(CardType.NUMBER,number_value)
        else:  # 33.333%概率运算符牌
            # 表示不同的运算符, 加减乘除和括号比例是3:3:3:2:1
            operator_types = [
                OperatorType.PLUS, OperatorType.MINUS, OperatorType.MULTIPLY,
                OperatorType.PLUS, OperatorType.MINUS, OperatorType.MULTIPLY,
                OperatorType.PLUS, OperatorType.MINUS, OperatorType.MULTIPLY,
                OperatorType.DIVIDE, OperatorType.DIVIDE,
                OperatorType.PLUS, OperatorType.MINUS, OperatorType.MULTIPLY,
                OperatorType.PLUS, OperatorType.MINUS, OperatorType.MULTIPLY,
                OperatorType.PLUS, OperatorType.MINUS, OperatorType.MULTIPLY,
                OperatorType.DIVIDE, OperatorType.DIVIDE,
                OperatorType.LEFTBRA, OperatorType.RIGHTBRA
            ]
            chosen_operator = random.choice(operator_types)
            return Card(CardType.OPERATOR, operator_type=chosen_operator)
    
    def generate_random_number_card(self):
        """生成随机数字卡牌"""
        number_value = random.randint(0, 13)
        return Card(CardType.NUMBER, number_value)
    
    def generate_random_op_card(self):
        """生成随机运算符卡牌"""
        operator_types = [
            OperatorType.PLUS, OperatorType.MINUS, OperatorType.MULTIPLY,
            OperatorType.PLUS, OperatorType.MINUS, OperatorType.MULTIPLY,
            OperatorType.PLUS, OperatorType.MINUS, OperatorType.MULTIPLY,
            OperatorType.DIVIDE,OperatorType.DIVIDE, 
        ]   # 加减乘除比例是3:3:3:2
        chosen_operator = random.choice(operator_types)
        return Card(CardType.OPERATOR, operator_type=chosen_operator)
        
    def generate_skill_card(self, skill_type: SkillType) -> Card:
        """生成技能牌"""
        return Card(CardType.SKILL, skill_type = skill_type)
    
    def calculate_expression(self, cards: List[Card]) -> Optional[int]:
        """计算表达式结果"""
        try:
            # 基本语法检查
            if not self.is_valid_expression(cards):
                return None
            
            # 构建表达式字符串
            tokens = []
            for card in cards:
                if card.card_type == CardType.NUMBER:
                    tokens.append(str(card.value))
                elif card.card_type == CardType.OPERATOR:
                    if card.operator_type == OperatorType.PLUS:
                        tokens.append('+')
                    elif card.operator_type == OperatorType.MINUS:
                        tokens.append('-')
                    elif card.operator_type == OperatorType.MULTIPLY:
                        tokens.append('*')
                    elif card.operator_type == OperatorType.DIVIDE:
                        tokens.append('/')
                    elif card.operator_type == OperatorType.LEFTBRA:
                        tokens.append('(')
                    elif card.operator_type == OperatorType.RIGHTBRA:
                        tokens.append(')')
            
            # 使用安全的表达式计算
            result = self.simple_evaluate(tokens)
            
            if result is not None:
                # 1. 如果结果是非常小的负数 (如 -1e-16), 修正为 0
                if -1e-3 < result < 0:
                    result = 0.0
                
                # 2. 只有修正后 >= 0 且有效才返回
                if result >= 0:
                    # 使用 round 处理 23.99999 -> 24
                    # 避免直接 int(23.999) -> 23 的悲剧
                    nearest_int = round(result)
                    if abs(result - nearest_int) < 1e-3:
                        return nearest_int
                    else:
                        return round(result, 2)
            
        except (ZeroDivisionError, ValueError, TypeError):
            return None
        
        return None
    
    def simple_evaluate(self, tokens: List) -> Optional[float]:
        """
        简化版的安全计算，使用递归下降解析
        适用于不太复杂的表达式
        """
        tokens = tokens.copy()
        index = [0]  # 使用列表以便在递归中修改
        
        def parse_expression():
            result = parse_term()
            while index[0] < len(tokens) and tokens[index[0]] in ('+', '-'):
                op = tokens[index[0]]
                index[0] += 1
                term = parse_term()
                if op == '+':
                    result += term
                else:
                    result -= term
            return result
        
        def parse_term():
            result = parse_factor()
            while index[0] < len(tokens) and tokens[index[0]] in ('*', '/'):
                op = tokens[index[0]]
                index[0] += 1
                factor = parse_factor()
                if op == '*':
                    result *= factor
                else:
                    if factor == 0:
                        return None
                    result /= factor
            return result
        
        def parse_factor():
            if index[0] >= len(tokens):
                raise ValueError("Not complete expression")
            
            token = tokens[index[0]]
            if token == '(':
                index[0] += 1
                result = parse_expression()
                if index[0] >= len(tokens) or tokens[index[0]] != ')':
                    raise ValueError("Brackets not match")
                index[0] += 1
                return result
            
            try:
                # 如果是数字字符串，转换并返回
                result = float(token)
                index[0] += 1
                return result
            except (ValueError, TypeError):
                # 如果不是数字，抛异常
                raise ValueError(f"Unexpected token: {token}")
        
        try:
            result = parse_expression()
            if index[0] != len(tokens):
                raise ValueError("Not complete expression")
            return result
        except (IndexError, ValueError, ZeroDivisionError):
            return None

    def is_valid_expression(self, cards: List[Card]) -> bool:
        """严格验证卡牌序列 - 不允许数字和括号相邻"""
        if len(cards) < 3:
            return False
        
        # 使用状态机进行验证
        bracket_count = 0  # 括号计数
        expect_operand = True  # 期望操作数（数字或左括号）
        
        for i, card in enumerate(cards):
            if card.card_type == CardType.NUMBER:
                # 当前是数字：必须期望操作数
                if not expect_operand:
                    return False
                expect_operand = False  # 数字后期望运算符或右括号
                
            elif card.card_type == CardType.OPERATOR:
                if card.operator_type == OperatorType.LEFTBRA:
                    # 左括号：必须期望操作数，增加括号计数
                    if not expect_operand:
                        return False
                    bracket_count += 1
                    # 括号内重新期望操作数
                    expect_operand = True
                    
                elif card.operator_type == OperatorType.RIGHTBRA:
                    # 右括号：不能期望操作数，减少括号计数
                    if expect_operand or bracket_count == 0:
                        return False
                    bracket_count -= 1
                    # 右括号后期望运算符或右括号
                    expect_operand = False
                    
                else:  # 加减乘除运算符
                    # 运算符：不能期望操作数（即不能连续运算符）
                    if expect_operand:
                        return False
                    # 运算符后期望操作数
                    expect_operand = True
            
            # 检查括号计数是否有效
            if bracket_count < 0:
                return False
        
        # 最终检查：括号必须匹配，不能以运算符结尾
        return bracket_count == 0 and not expect_operand
    
    def check_skill_triggers(self, result: int) -> str:
        """检查技能触发条件，返回触发的技能描述"""
        if result is None:
            return ""
        
        skill_message = ""
        #检查1
        if result == 1:
            self.current_player.add_card(self.generate_skill_card(SkillType.PIERCE))
            skill_message = f" ONE skill triggered! Got pierce card"
        #检查0
        if result ==0:
            self.current_player.add_card(self.generate_skill_card(SkillType.RUIN))
            skill_message = f" ZERO skill triggered! Got ruin card"
            
        # 检查阶乘
        for i in range(2, 13):
            if math.factorial(i) == result:
                self.current_player.add_card(self.generate_skill_card(SkillType.HEAL))
                skill_message = f" Factorial skill triggered! Got heal card"
                
                break
        
        # 检查cube
        if result > 0:
            cbrt_result = int(round(result ** (1/3)))
        else:
            cbrt_result = 20000
        if cbrt_result ** 3 == result and result!= 0 and result != 1:
            self.current_player.add_card(self.generate_skill_card(SkillType.STEAL))
            skill_message = f" Cube skill triggered! Got steal card"
        
        # 检查square
        if result > 0:
            sqrt_result = int(math.sqrt(result))
        else:
            sqrt_result = 20000
        if sqrt_result ** 2 == result and result != 0 and result != 1:
            self.current_player.add_card(self.generate_skill_card(SkillType.DRAW))
            skill_message = f" Square skill triggered! Got Draw card"
            
        # 检查24点
        if result == 24:
            self.current_player.add_card(self.generate_skill_card(SkillType.SHIELD))
            self.current_player.add_card(self.generate_skill_card(SkillType.SHIELD))
            skill_message = f" 24-point skill triggered! Got 2 shield cards"
        
        return skill_message
    
    def use_skill_card(self, skill_card: Card):
        """使用技能牌"""
        if skill_card.skill_type == SkillType.HEAL:
            # 恢复20点生命值并抽1张牌
            self.current_player.hp = min(120, self.current_player.hp + 20)
            self.current_player.add_card(self.generate_random_card())
            
        elif skill_card.skill_type == SkillType.PIERCE:
            # 直接破坏对方所有护盾
            other_player = self.player2 if self.current_player == self.player1 else self.player1
            other_player.shield_count =0
        
        elif skill_card.skill_type == SkillType.RUIN:
            # 随机从对方手牌中毁掉3张
            other_player = self.player2 if self.current_player == self.player1 else self.player1
            available_cards = [i for i, count in enumerate(other_player.hand) if count > 0]
            for _ in range(min(3, len(available_cards))):
                if available_cards:
                    card_index = random.choice(available_cards)
                    other_player.hand[card_index] -= 1
                    available_cards = [i for i, count in enumerate(other_player.hand) if count > 0]


        elif skill_card.skill_type == SkillType.STEAL:
            # 随机从对方手牌中偷牌，若手牌差距小于等于6张则偷2张，否则偷1张
            other_player = self.player2 if self.current_player == self.player1 else self.player1
            current_hand_count = sum(self.current_player.hand)
            other_hand_count = sum(other_player.hand)
            steal_count = 2 if current_hand_count - other_hand_count <= 6 else 1
            
            available_cards = [i for i, count in enumerate(other_player.hand) if count > 0]
            for _ in range(min(steal_count, len(available_cards))):
                if available_cards:
                    card_index = random.choice(available_cards)
                    other_player.hand[card_index] -= 1
                    # 给当前玩家添加对应的卡牌
                    if card_index <= 13:
                        stolen_card = Card(CardType.NUMBER, value=card_index)
                    else:
                        operator_type = OperatorType(card_index)
                        stolen_card = Card(CardType.OPERATOR, operator_type=operator_type)
                    self.current_player.add_card(stolen_card)
                    available_cards = [i for i, count in enumerate(other_player.hand) if count > 0]

        elif skill_card.skill_type == SkillType.DRAW:
            for _ in range(2):
                self.current_player.add_card(self.generate_random_number_card())
            self.current_player.add_card(self.generate_random_op_card())
            
        elif skill_card.skill_type == SkillType.SHIELD:
            self.current_player.shield_count += 1
            
        self.current_player.remove_card(skill_card)
    
    def switch_player(self):
        """切换玩家 结束一次连续操作"""
        # 返还选中的卡牌
        for card in self.selected_cards:
            self.current_player.add_card(card)
        self.selected_cards = []
        self.symbolic_executor.reset()

        self.current_player.is_active = False
        
        self.current_player = self.player2 if self.current_player == self.player1 else self.player1
        self.current_player.is_active = True
        self.selected_cards = []
        self.calculation_result = None
        self.continuous_operations = 0
        self.nonaction_times = 0
        self.noneffect_times = 0
        
        # 检查切换到的玩家是否已结束本轮
        if (self.current_player == self.player1 and self.player1_round_end) or \
           (self.current_player == self.player2 and self.player2_round_end):
            # 如果切换到的玩家已结束本轮，继续切换
            self.switch_player()
        else:
            self.message = f"{self.current_player.name}'s turn"
    
    def end_current_round(self):
        # 返还选中的卡牌
        for card in self.selected_cards:
            self.current_player.add_card(card)
        self.selected_cards = []
        self.symbolic_executor.reset()
        self.nonaction_times = 0
        self.noneffect_times = 0

        """当前玩家结束本轮"""
        if self.current_player == self.player1:
            self.player1_round_end = True
        else:
            self.player2_round_end = True
        
        # 记录先结束本轮的玩家
        if self.first_to_end_round is None:
            self.first_to_end_round = self.current_player

        
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
            else:
                # 对手也已结束本轮，直接结束轮次
                self.end_round()
    
    def end_round(self):
        """结束当前轮次 双方都结束本轮后调用"""
        # 下一轮先手权给当前轮次先结束本轮的玩家
        if self.first_to_end_round is not None:
            self.current_player = self.first_to_end_round
            # 后手补偿2张牌
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
        self.newcardsnum = 5 + int(self.round_number / 2)  # 每2轮增加1张新牌
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


class SymbolicExecutor:
    """可微分符号执行器：追踪表达式求值过程"""
    
    def __init__(self):
        self.reset()
        self.pending_bracket_depth = 0 
    
    def reset(self):
        self.value_stack = []  # 数值栈
        self.op_stack = []     # 运算符栈（含括号）
        self.trace = []        # 执行轨迹
        self.bracket_depth = 0
        self.pending_bracket_depth = 0 
    
    def execute_step(self, card: Card, step_idx: int) -> Tuple[float, int, bool]:
        """
        单步执行，返回：(当前栈顶值, 括号深度, 是否有效)
        """
        current_value = self.value_stack[-1] if self.value_stack else 0.0
        
        if card.card_type == CardType.NUMBER:
            # 数字：压入value栈
            self.value_stack.append(float(card.value))
            
            # 检查是否可以计算（如 3 + 5）
            if len(self.value_stack) >= 2 and self.op_stack and self.op_stack[-1] != '(':
                self._pop_calc()
            
        elif card.card_type == CardType.OPERATOR:
            if card.operator_type == OperatorType.LEFTBRA:
                self.bracket_depth += 1
                self.op_stack.append('(')
                self.pending_bracket_depth += 1
            elif card.operator_type == OperatorType.RIGHTBRA:
                self.bracket_depth -= 1
                self.pending_bracket_depth -= 1
                if self.bracket_depth < 0:
                    return -999, 0, False  # 无效
                
                # 计算到左括号为止
                while self.op_stack and self.op_stack[-1] != '(':
                    self._pop_calc()
                if self.op_stack and self.op_stack[-1] == '(':
                    self.op_stack.pop()  # 弹出左括号
            else:
                # 运算符：处理优先级
                op = self._op_to_str(card.operator_type)
                while (self.op_stack and 
                       self._op_priority(self.op_stack[-1]) >= self._op_priority(op)):
                    self._pop_calc()
                self.op_stack.append(op)
        
        # 更新当前值（执行后栈顶）
        if self.pending_bracket_depth == 0:
            current_value = self.value_stack[-1] if self.value_stack else 0.0
            is_valid = True
        else:
            current_value = None  # 挂起状态，结果不确定
            is_valid = False  # 不可评估
            
        # 记录trace
        self.trace.append((step_idx, current_value, self.bracket_depth, is_valid))
        
        return current_value, self.bracket_depth, is_valid
    
    def _pop_calc(self):
        """弹出计算：取两个数和运算符"""
        if len(self.value_stack) < 2:
            return
        
        b = self.value_stack.pop()
        a = self.value_stack.pop()
        op = self.op_stack.pop()
        
        try:
            result = self._apply_op(a, b, op)
            self.value_stack.append(result)
        except ZeroDivisionError:
            self.value_stack.append(-999)  # 标记无效
    
    def _op_to_str(self, op_type: OperatorType) -> str:
        mapping = {
            OperatorType.PLUS: '+',
            OperatorType.MINUS: '-',
            OperatorType.MULTIPLY: '*',
            OperatorType.DIVIDE: '/'
        }
        return mapping.get(op_type, '')
    
    def _op_priority(self, op: str) -> int:
        if op in ['+', '-']: return 1
        if op in ['*', '/']: return 2
        return 0
    
    def _apply_op(self, a: float, b: float, op: str) -> float:
        if op == '+': return a + b
        if op == '-': return a - b
        if op == '*': return a * b
        if op == '/': 
            if b == 0: raise ZeroDivisionError
            return a / b
        return 0
    
    def get_trace(self) -> List[Tuple[int, float, int, bool]]:
        """返回执行轨迹：[(step_id, stack_top, depth, valid), ...]"""
        return self.trace
    
    def get_final_result(self) -> Optional[float]:
        """执行完成后获取最终结果"""
        if not self.trace:
            return None
        return self.trace[-1][1] if self.trace[-1][3] else None


# 在游戏初始化后运行测试
if __name__ == "__main__":   
    game = Game()


