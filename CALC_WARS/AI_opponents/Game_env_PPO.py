import pygame
import random
import math
from enum import Enum
from typing import List, Dict, Tuple, Optional
import os
import numpy as np

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
    __slots__ = ('card_type', 'value', 'skill_type', 'operator_type', 'used')  # 减少__dict__开销
    
    def __init__(self, card_type: CardType, 
                 value: Optional[int] = None,
                 skill_type: Optional[SkillType] = None,
                 operator_type: Optional[OperatorType] = None):
        self.card_type = card_type
        self.value = value
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
        self.hp = 100
        self.hand = [0] * 20# 手牌（数字和运算符）
        self.skill_cards = [0] * 6# 技能卡
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
        self.yellow_zone = random.sample(range(24, 37),2)
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

class CurriculumTarget(Target):
    """课程学习靶子：渐进式随机化"""
    def __init__(self, curriculum_stage=0):
        self.stage = curriculum_stage
        self._randomize()
    
    def _randomize(self):
        """根据stage设置靶区"""
        if self.stage == 0:
            # 阶段0：完全固定（用于记忆）
            self.red_zone = 37
            self.yellow_zone = [30, 35]
            self.blue_zone = [11, 12, 20, 9]
        elif self.stage == 1:
            # 阶段1：红区随机，黄蓝区低方差
            self.red_zone = random.choice([37, 41, 43])
            self.yellow_zone = random.sample(range(30, 37), 2)
            self.blue_zone = random.sample(range(7, 23), 3) + random.sample([4,9,16], 1)
        else:
            # 阶段2：完全随机
            super().__init__()
    
    def advance_stage(self):
        """提升课程阶段"""
        self.stage = min(self.stage + 1, 2)
        self._randomize()
    

class Game:
    def __init__(self, use_curriculum=False):
        self.seq_len = 9
        self.player1 = Player("Player1")
        self.player2 = Player("Player2")
        self.current_player = self.player1
        self.target = Target()
        self.game_phase = "initial_deal"  # initial_deal, playing, round_end
        self.round_number = 1
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
        self.info = {"message": ""}

        self.nonaction_times = 0

        self.symbolic_executor = SymbolicExecutor()
        self.trainer = None  # 将由TrainingManager注入
        # 初始化游戏
        self.initial_deal()

        self._state_buffer = np.zeros(131, dtype=np.float32)
        self._mask_buffer = np.zeros(29, dtype=np.int8)
        self._zones_buffer = np.zeros(7, dtype=np.float32)
        
        # 预计算常量
        self._target_weights = np.array([3.0, 1.5, 1.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

        # 课程学习相关
        self.use_curriculum = use_curriculum
        self.curriculum_stage = 0  # 记录当前stage

        # 根据开关选择靶子类型
        if use_curriculum:
            self.target = CurriculumTarget(self.curriculum_stage)
        else:
            self.target = Target()


    def reset(self):
        self.player1.hand = [0] * 20
        self.player2.hand = [0] * 20
        self.player1.skill_cards = [0] * 6
        self.player2.skill_cards = [0] * 6

        self.player1.hp = 100
        self.player2.hp = 100
        
        # 同时重置护盾
        self.player1.shield_count = 0
        self.player2.shield_count = 0

        self.game_phase = "initial_deal"  # initial_deal, playing, round_end
        self.round_number = 1
        self.selected_cards = []
        self.calculation_result = None
        self.done = False
        self.player1_round_end = False  # 玩家1是否结束本轮
        self.player2_round_end = False  # 玩家2是否结束本轮
        self.continuous_operations = 0  # 连续操作次数
        self.first_to_end_round = None  # 当前轮次先结束本轮的玩家

        self.step_count = 0
        self.leftright_unbal_bra = 0

        self.nonaction_times = 0

        self.symbolic_executor.reset()
        # 重新发牌
        self.initial_deal()

        # 重置靶子时保持当前课程阶段
        if self.use_curriculum:
            # 保持stage不变，重新随机化（对stage1和stage2有效）
            self.target = CurriculumTarget(self.curriculum_stage)
        else:
            self.target = Target()

        return self._get_state()
    
    def log_expr_stat(self, key: str, value: float):
        """向训练管理器记录统计信息（如果存在）"""
        if self.trainer is not None and hasattr(self.trainer, 'expr_stats'):
            self.trainer.expr_stats[key].append(value)

    def set_curriculum_stage(self, stage: int):
        """外部接口：设置课程学习阶段"""
        if not self.use_curriculum:
            return
        
        self.curriculum_stage = stage
        # 立即应用新stage
        self.target = CurriculumTarget(stage)
        print(f"  Game target updated to stage {stage}")

    
    def _get_state(self):
        """状态生成向量化，减少Python循环"""
        state = self._state_buffer
        state.fill(0.0)
        
        opponent = self.player2 if self.current_player == self.player1 else self.player1
        
        # 批量赋值
        state[:20] = np.array(self.current_player.hand, dtype=np.float32) / 7.0
        state[20:40] = np.array(opponent.hand, dtype=np.float32) / 7.0
        state[40:46] = np.array(self.current_player.skill_cards, dtype=np.float32) / 5.0
        state[46:52] = np.array(opponent.skill_cards, dtype=np.float32) / 5.0
        
        # 标量特征
        state[52] = self.current_player.hp / 100.0
        state[53] = opponent.hp / 100.0
        state[54] = self.current_player.shield_count / 6.0
        state[55] = opponent.shield_count / 6.0
        state[56] = self.round_number / 6.0
        state[57] = self.round_number ** 2 / 36.0
        state[58] = self.step_count ** 1.2 * 4.0 / 5000.0
        state[59] = self.continuous_operations / 5.0
        state[60] = (state[:20].sum() - state[20:40].sum())/10.0 #手牌差距
        state[61] = (self.current_player.hp - opponent.hp) / 100.0 #血量差距
        state[62] = 1.0 if self.current_player == self.player1 else 0.0
        
        if self.current_player == self.player1:
            state[63] = 1.0 if self.player1_round_end else 0.0
            state[64] = 1.0 if self.player2_round_end else 0.0
        else:
            state[63] = 1.0 if self.player2_round_end else 0.0
            state[64] = 1.0 if self.player1_round_end else 0.0
    
        
        # 目标区域（批量操作）
        zones = self._zones_buffer
        zones[0] = self.target.red_zone
        zones[1:3] = self.target.yellow_zone[:2]
        zones[3:7] = self.target.blue_zone[:4]
        state[65:72] = zones / 60.0
        
        # 距离计算向量化
        trace = self.symbolic_executor.trace
        if trace and trace[-1][1] is not None:
            current_partial = float(trace[-1][1])  # 有效部分值
        else:
            current_partial = 0.0  # 无效表达式设为0
        distances = np.abs(current_partial - zones) / 60.0
        state[72:79] = self._target_weights * np.exp(-distances / 10.0)
        
        # 选择序列（批量填充）
        seq_len = min(len(self.selected_cards), self.seq_len)
        for i, card in enumerate(self.selected_cards[:seq_len]):
            if card.card_type == CardType.NUMBER:
                state[79 + i*2] = card.value / 20.0
                state[79 + i*2 + 1] = 0.0  # 数字类型
            else:  # OPERATOR
                state[79 + i*2] = card.operator_type.value / 20.0
                
                # **正确还原运算符子类型！**
                if card.operator_type in [OperatorType.PLUS, OperatorType.MINUS]:
                    state[79 + i*2 + 1] = 1.0 / 4.0  # 加减 → 0.25
                elif card.operator_type in [OperatorType.MULTIPLY, OperatorType.DIVIDE]:
                    state[79 + i*2 + 1] = 2.0 / 4.0  # 乘除 → 0.5
                else:  # 括号
                    state[79 + i*2 + 1] = 3.0 / 4.0  # 括号 → 0.75
        
        # 10. PAD标记
        for i in range(seq_len, self.seq_len):
            state[79 + i*2] = 1.0        # PAD索引: 20.0/20.0 = 1.0
            state[79 + i*2 + 1] = 0.2    # 空类型: 4.0/20.0 = 0.2
        
        # 符号特征（批量填充）
        trace_len = min(len(trace), 10)
        for i, t in enumerate(trace[:trace_len]):
            state[97 + i*3] = (t[1] / 100.0) if t[1] is not None else -1.0
            state[97 + i*3 + 1] = t[2] / 5.0
            state[97 + i*3 + 2] = 1.0 if t[3] else 0.0  # validity (valid=1.0, invalid=0.0)
        
        # PAD标记（批量）
        state[97 + trace_len*3:127] = -2.0


        state[127] = self.nonaction_times / 3.0
        state[128] = (self.nonaction_times  ** 2) / 9.0
        total_actions = self.continuous_operations + self.nonaction_times + 1
        state[129] = self.nonaction_times / total_actions
        state[130] = 1.0 if self.done else 0.0  # 游戏结束标志
        
        return state
    
    def get_valid_actions_mask(self):
        # 动作空间设计：
        # 0-13: 选择手牌中的数字牌
        # 14-19: 选择手牌中的符号牌
        # 20-25: 使用技能牌
        # 26: 确认出牌（技能牌无需确认）
        # 27: 结束本回合
        # 28：结束本轮
        # 总共29种动作

        mask = self._mask_buffer
        mask.fill(0)
        
        hand = self.current_player.hand
        skills = self.current_player.skill_cards
        
        # 向量化赋值
        np.less(0, hand, out=mask[:20])  # hand > 0
        np.less(0, skills, out=mask[20:26])  # skills > 0
        
        # 快速路径：空序列
        if not self.selected_cards:
            mask[14:18] = 0
            mask[19] = 0
        else:
            # 增量检查：只分析最后一张牌
            last_card = self.selected_cards[-1]
            if last_card.card_type == CardType.NUMBER:
                mask[:14] = 0
                mask[18] = 0
            elif last_card.card_type == CardType.OPERATOR:
                if last_card.operator_type == OperatorType.LEFTBRA:
                    mask[14:20] = 0
                elif last_card.operator_type == OperatorType.RIGHTBRA:
                    mask[:14] = 0
                else:
                    mask[14:18] = 0
                    mask[19] = 0
            
            # 除零保护：只检查最后两张牌
            if len(self.selected_cards) >= 2:
                second_last = self.selected_cards[-2]
                if (second_last.card_type == CardType.NUMBER and 
                    last_card.card_type == CardType.OPERATOR and 
                    last_card.operator_type == OperatorType.DIVIDE):
                    mask[0] = 0  # 禁用数字0
        
        # 括号平衡检查（O(1)复杂度）
        leftbra_count = sum(1 for c in self.selected_cards 
                           if c.card_type == CardType.OPERATOR and 
                           c.operator_type == OperatorType.LEFTBRA)
        rightbra_count = sum(1 for c in self.selected_cards 
                            if c.card_type == CardType.OPERATOR and 
                            c.operator_type == OperatorType.RIGHTBRA)
        
        if leftbra_count <= rightbra_count:
            mask[19] = 0
        
        # 检查手牌中剩余括号 + 已选中的括号总数
        if (self.current_player.hand[18] + leftbra_count) == 0 or (self.current_player.hand[19] + rightbra_count) == 0:
            mask[18:20] = 0
        
        # 确认按钮逻辑
        min_len = 3 
        mask[26] = 1 if len(self.selected_cards) >= min_len else 0
        
        # 长度限制
        if len(self.selected_cards) >= 9:
            mask[:26] = 0
        
        mask[27] = 0 if (self.player1_round_end or self.player2_round_end) else 1
        mask[28] = 1
        
        if self.selected_cards:
            mask[27:29] = 0
        
        # 使用位运算快速检查
        if not self.Operatability_check():
            mask[:27] = 0
        
        if self.nonaction_times >= 4:
            mask[:27] = 0
        
        # 快速any检查
        if not np.any(mask):
            mask[27:29] = 1
        
        return mask
    
    def Operatability_check(self):
        ava_op = 0
        ava_num = 0
        # 统计手牌中的数字牌（索引0-13）
        for i in range(14):
            ava_num += self.current_player.hand[i]
        
        # 统计手牌中的运算符牌（索引14-17）
        for i in range(14, 18):
            ava_op += self.current_player.hand[i]
        
        # 统计已选中的卡牌
        for card in self.selected_cards:
            if card.card_type == CardType.NUMBER:
                ava_num += 1
            elif card.card_type == CardType.OPERATOR and card.operator_type in [
                OperatorType.PLUS, OperatorType.MINUS, OperatorType.MULTIPLY, OperatorType.DIVIDE
            ]:
                ava_op += 1

        return ava_num >= 2 and ava_op >= 1
    

    def step(self, action):
        """
        RL环境的核心方法
        action: 整数，表示选择的动作
        返回: (next_state, reward, done, info)
        """
        opponent = self.player2 if self.current_player == self.player1 else self.player1
        reward = 0
        done = False
        self.step_count += 1

        time_penalty = -1.032 ** (self.step_count / 5.0) + 1 # 随时间递增
        time_penalty = 0
        
    
        self.brackert_count = 0
        self.info = {"message": ""}

        valid_mask = self.get_valid_actions_mask()

        if action < 0 or action >= len(valid_mask):
            reward = -50.0
            self.info["message"] = f"ACTION OUT OF BOUNDS: {action}"
            self.nonaction_times += 1
            return self._get_state(), reward, self.done, self.info
        
        # 掩码检查
        if valid_mask[action] == 0:
            reward = -8.0  # 标准非法动作惩罚
            self.info["message"] = f"MASK VIOLATION: Action {action}"
            self.nonaction_times += 1
            
            # 关键：直接返回，不执行任何游戏逻辑
            return self._get_state(), reward, self.done, self.info
        
        # 检查游戏是否已经结束
        if self.done:
            return self._get_state(), 0, True, {"message": "Game already finished"}
        
        # 解析动作
        if action < 20 :  # 选择手牌
            if self.current_player.hand[action] > 0:
                if action <= 13:  # 数字卡
                    card = Card(CardType.NUMBER, value=action)
                    self.current_player.hand[action] -= 1

                else:  # 运算符卡
                    operator_type = OperatorType(action)
                    card = Card(CardType.OPERATOR, operator_type=operator_type)
                    self.current_player.hand[action] -= 1

                    if action in [18,19]:  #鼓励AI学会括号
                        reward += 1.0

                    leftbra_count = sum(1 for card in self.selected_cards 
                        if card.card_type == CardType.OPERATOR and card.operator_type == OperatorType.LEFTBRA)
                    
                    rightbra_count = sum(1 for card in self.selected_cards 
                        if card.card_type == CardType.OPERATOR and card.operator_type == OperatorType.RIGHTBRA)
                    
                    leftright_unbal_bra = leftbra_count - rightbra_count

                    if leftright_unbal_bra > 0 and action == 19: #如果左括号匹配上右括号，大胜利
                        reward += 3.0

                    trace = self.symbolic_executor.trace
                    if len(trace) >= 2:  # 至少需要两步才能比较
                        # 提取当前和上一个中间值
                        curr_entry = trace[-1]  # (step_idx, value, depth, valid)
                        prev_entry = trace[-2]
                        
                        curr_value = curr_entry[1] if curr_entry[1] is not None else None
                        prev_value = prev_entry[1] if prev_entry[1] is not None else None
                        
                        if curr_value is not None and curr_value >= 0:
                            # 计算到各靶区的距离
                            red_dist = abs(self.target.red_zone - curr_value)
                            yellow_dists = [abs(y - curr_value) for y in self.target.yellow_zone]
                            blue_dists = [abs(b - curr_value) for b in self.target.blue_zone]
                            curr_min_dist = min(red_dist, *yellow_dists, *blue_dists)
                            
                            # 1. 距离改进奖励（相比上一步）
                            if prev_value is not None:
                                red_dist = abs(self.target.red_zone - curr_value)
                                yellow_dists = [abs(y - curr_value) for y in self.target.yellow_zone]
                                blue_dists = [abs(b - curr_value) for b in self.target.blue_zone]
                                prev_min_dist = min(red_dist, *yellow_dists, *blue_dists)
                                
                                if curr_min_dist < prev_min_dist:
                                    reward += (prev_min_dist - curr_min_dist) * 3.0  # 每单位改进+3
                            
                            # 2. 接近靶区的额外奖励
                            if curr_min_dist < 5.0:
                                reward += (5.0 - curr_min_dist) * 8.0  # 在5距离内高额奖励
                            
                            # 3. 深度奖励：括号内计算有效
                            if curr_entry[2] > 0 and prev_value is not None and curr_value > prev_value:
                                reward += curr_entry[2] * 2.0


                self.selected_cards.append(card)    #转换成表达式再把所有大于13的牌变为符号
                if card.card_type == CardType.OPERATOR and card.operator_type == OperatorType.RIGHTBRA:
                    # 在SymbolicExecutor.execute_step中已经存储了_last_bracket_multiplier
                    multiplier = getattr(self.symbolic_executor, '_last_bracket_multiplier', None)
                    if multiplier and multiplier > 1.5:
                        reward += multiplier * 5.0
                        self.log_expr_stat('bracket_effectiveness', multiplier)
                    else:
                        self.log_expr_stat('bracket_effectiveness', 1.0)
                self.symbolic_executor.execute_step(card, len(self.selected_cards) - 1)
                self.info["message"] = f"Selected card: {card}"

            else:
                reward = -2.0 # 选择不存在的卡牌惩罚
                self.info["message"] = "Card not available"
            pass
        
        elif action < 26 :  # 使用技能牌
            skill_index = action - 20
            if self.current_player.skill_cards[skill_index] > 0:
                skill_type = SkillType(skill_index)
                skill_card = Card(CardType.SKILL, skill_type=skill_type)
                self.use_skill_card(skill_card)
                self.info["message"] = f"Used skill: {skill_type.name}"
                reward += 30.0 # 使用技能奖励
            else:
                reward = -2.0 # 选择不存在的卡牌惩罚
                self.info["message"] = "Card not available"
            pass
        
        elif action == 27:  # 结束当前回合
            reward += time_penalty

            if self.continuous_operations == 0:
                if self.Operatability_check():
                    reward += -12.0
                else:
                    reward += -6.0
            if self.selected_cards:
                reward += -10.0
            if not self.player1_round_end and not self.player2_round_end:
                # 补牌
                for _ in range(2):
                    makeup_card = self.generate_random_card()
                    self.current_player.add_card(makeup_card)
                self.switch_player()
                self.info["message"] = "Ended turn" 
            else:
                self.info["message"] = "End not legal"
                reward += -10.0
                self.nonaction_times += 1
            pass
        
        elif action == 28:  # 结束本轮
            reward += time_penalty
            reward += -10.0 * self.round_number ** 0.8

            if self.continuous_operations == 0:
                if self.Operatability_check():
                    reward += -12.0
                else:
                    reward += -6.0
            if self.selected_cards:
                reward += -10.0
            self.end_current_round()
            self.info["message"] = "Ended round"
        

        elif action == 26:  # 确认出牌
            reward += time_penalty
            #为括号设计特别机制 用对了大奖励用错了小惩罚
            self.brackert_count = sum(1 for c in self.selected_cards 
                        if c.card_type == CardType.OPERATOR and \
                            (c.operator_type == OperatorType.RIGHTBRA or c.operator_type == OperatorType.LEFTBRA))
            if len(self.selected_cards) >= 3:
                result = self.symbolic_executor.get_final_result()
                if result is not None and result >= 0:
                    self.nonaction_times = 0
                    self.calculation_result = result
                    damage = self.target.get_damage(result)
                    actual_damage = opponent.take_damage(damage)
                    
                    # 奖励基于造成的伤害
                    reward += 10.0
                    reward += actual_damage * 4.0   #伤害奖励

                    cards_used = len(self.selected_cards)
                    efficiency = actual_damage / max(cards_used, 1)
                    reward += efficiency * 2.0  # 牌效奖励
                    reward += (cards_used - 3.0) * 5.0

                    red_dist = abs(self.target.red_zone - result)
                    reward += np.exp(-red_dist / 5.0) * 50.0  # 距离5以内有显著奖励

                    if result >= 20:
                        reward += max(0.0, (result - 20.0) ** 0.1 * 4.0 )

                    mul_num = 0
                    mius_add_num = 0
                    for card in self.selected_cards:
                        if card.card_type == CardType.OPERATOR:
                            if card.operator_type == OperatorType.MULTIPLY:
                                mul_num += 1
                            if card.operator_type == OperatorType.PLUS or card.operator_type == OperatorType.MINUS:
                                mius_add_num += 1
                    if mul_num >= 1 and mius_add_num >= 1:
                        reward += 10.0
                    
                    # 检查技能触发
                    skill_message = self.check_skill_triggers(result)
                    if skill_message:
                        reward += 30.0  # 技能触发奖励
                        self.info["skill_triggered"] = skill_message

                    reward += 2.0 * self.brackert_count
    
                    self.selected_cards = []
                    self.symbolic_executor.reset()
                    self.continuous_operations += 1
                    self.info["message"] = f"Played cards, result: {result}, damage: {actual_damage}"
                else:
                    reward = -8.0 # 无效表达式惩罚
                    self.info["message"] = "Invalid expression"


                    # 返还选中的卡牌
                    for card in self.selected_cards:
                        self.current_player.add_card(card)
                    self.nonaction_times += 1
                    self.selected_cards = []
            else:
                reward = -20.0 # 卡牌数量不足惩罚
                self.info["message"] = "Need at least 3 cards"
                # 返还选中的卡牌
                for card in self.selected_cards:
                    self.current_player.add_card(card)
                self.nonaction_times += 1
                self.selected_cards = []
        
        else:
            reward = -20.0 # 无效动作惩罚
            self.info["message"] = "Invalid action"
            self.nonaction_times += 1

        # 添加连续操作奖励/惩罚
        if self.continuous_operations > 0:
            reward +=  1.0 * self.continuous_operations

        reward += - 5.0 * self.nonaction_times

        # 检查游戏是否结束
        if self.player1.hp <= 0 or self.player2.hp <= 0:
            self.done = True
            if self.current_player.hp > 0:
                reward = 1600  - 1.4 ** (self.step_count / 25.0) # 获胜奖励 鼓励速胜
            else:
                reward = -800.0  # 失败惩罚
        
        return self._get_state(), reward, self.done, self.info

    def bracket_health(self):
        """验证卡牌健康度"""
        if len(self.selected_cards) < 3:
            return 0.7

        health = 0
        left_stack = []  # 存储未匹配的左括号位置
        illegal_rights = 0  # 非法右括号计数
    
        for i, c in enumerate(self.selected_cards):
            if c.card_type == CardType.OPERATOR:
                if c.operator_type == OperatorType.LEFTBRA:
                    left_stack.append(i)  # 压栈
                
                elif c.operator_type == OperatorType.RIGHTBRA:
                    if left_stack:
                        left_stack.pop()  # 成功匹配，出栈
                    else:
                        illegal_rights += 1  # 灾难：右括号无左匹配
        
        unmatched_lefts = len(left_stack)  # 轻微错误：左括号未关闭
        
        health += unmatched_lefts * 0.2 + 0.75 * illegal_rights

        return health

    def initial_deal(self):
        """初始发牌"""
        for i in range(7):
            self.player1.add_card(self.generate_random_card())
            self.player2.add_card(self.generate_random_card())
            
        # 随机决定先手
        if random.random() < 0.5:
            self.current_player = self.player1
        else:
            self.current_player = self.player2
        
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
            # 表示不同的运算符, 加减乘除和括号比例是3:3:3:2:2:1
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
        
    def generate_skill_card(self, skill_type: SkillType) -> Card:
        """生成技能牌"""
        return Card(CardType.SKILL, skill_type = skill_type)
    
    def calculate_expression(self, cards: List[Card]) -> Optional[int]:
        """高性能表达式计算：复用SymbolicExecutor逻辑"""
        executor = SymbolicExecutor()  # 轻量级临时执行器
        for i, card in enumerate(cards):
            executor.execute_step(card, i)
        
        result = executor.get_final_result()
        if result is not None and result >= 0:
            return int(result) if result == int(result) else round(result, 2)
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
        """O(n)状态机验证，无递归"""
        if len(cards) < 3:
            return False
        
        bracket_count = 0
        expect_operand = True
        
        for card in cards:
            if card.card_type == CardType.NUMBER:
                if not expect_operand:
                    return False
                expect_operand = False
            elif card.card_type == CardType.OPERATOR:
                op_type = card.operator_type
                if op_type == OperatorType.LEFTBRA:
                    if not expect_operand:
                        return False
                    bracket_count += 1
                elif op_type == OperatorType.RIGHTBRA:
                    if expect_operand or bracket_count == 0:
                        return False
                    bracket_count -= 1
                else:  # 加减乘除
                    if expect_operand:
                        return False
                    expect_operand = True
                
                if bracket_count < 0:
                    return False
        
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
            self.current_player.hp = min(100, self.current_player.hp + 20)
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
            # 随机从对方手牌中偷牌，若手牌差距大于等于6张则偷2张，否则偷1张
            other_player = self.player2 if self.current_player == self.player1 else self.player1
            current_hand_count = sum(self.current_player.hand)
            other_hand_count = sum(other_player.hand)
            steal_count = 2 if current_hand_count - other_hand_count >= 6 else 1
            
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
           for _ in range(3):
            self.current_player.add_card(self.generate_random_card())
            
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
    """高性能符号执行器：迭代实现 + 状态复用"""
    
    # 预定义优先级和运算符映射（类属性，只创建一次）
    _OP_PRIORITY = {'+': 1, '-': 1, '*': 2, '/': 2, '(': 0}
    _OP_MAP = {
        OperatorType.PLUS: '+', OperatorType.MINUS: '-',
        OperatorType.MULTIPLY: '*', OperatorType.DIVIDE: '/'
    }
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        # 使用列表而非频繁创建新对象
        self.value_stack = []
        self.op_stack = []
        self.trace = []
        self.bracket_depth = 0
        self._last_state = (0.0, 0, True)  # 缓存最后状态，避免重复计算
    
    def execute_step(self, card: Card, step_idx: int) -> Tuple[float, int, bool]:
        """单步执行，使用迭代计算，性能提升5-10x"""
        if card.card_type == CardType.NUMBER:
            self.value_stack.append(float(card.value))
            self._try_calculate()  # 尝试计算
            
        elif card.card_type == CardType.OPERATOR:
            if card.operator_type == OperatorType.LEFTBRA:
                self.bracket_depth += 1
                self.op_stack.append('(')
            elif card.operator_type == OperatorType.RIGHTBRA:
                self.bracket_depth -= 1
                if self.bracket_depth < 0:
                    self._last_state = (-999.0, 0, False)
                    self.trace.append((step_idx, -999.0, 0, False))
                    return self._last_state
                
                # 批量计算到左括号
                while self.op_stack and self.op_stack[-1] != '(':
                    self._pop_calc()
                if self.op_stack and self.op_stack[-1] == '(':
                    self.op_stack.pop()
            else:
                # 处理运算符优先级
                op = self._OP_MAP[card.operator_type]
                while (self.op_stack and 
                       self._OP_PRIORITY[self.op_stack[-1]] >= self._OP_PRIORITY[op]):
                    self._pop_calc()
                self.op_stack.append(op)
        
        # 获取当前状态
        current_value = self.value_stack[-1] if self.value_stack else 0.0
        is_valid = self.bracket_depth >= 0 and not any(v is None for v in self.value_stack)
        
        # 复用状态对象，减少元组创建
        if (current_value != self._last_state[0] or 
            self.bracket_depth != self._last_state[1] or 
            is_valid != self._last_state[2]):
            self._last_state = (current_value, self.bracket_depth, is_valid)
        
        self.trace.append((step_idx, current_value, self.bracket_depth, is_valid))
        if (card.card_type == CardType.OPERATOR and card.operator_type == OperatorType.RIGHTBRA and len(self.trace) >= 2):
            # 找到匹配的左括号位置（进入括号前的状态）
            target_depth = self.bracket_depth + 1  # 当前在括号内，匹配前深度+1
            left_value = None
            
            for i in range(len(self.trace)-1, -1, -1):
                if self.trace[i][2] == target_depth:
                    left_value = self.trace[i][1]
                    break
            
            # 计算倍数并存储
            if left_value is not None and left_value > 0:
                current_value = self.value_stack[-1] if self.value_stack else None
                if current_value is not None and current_value > 0:
                    multiplier = current_value / left_value
                    self._last_bracket_multiplier = multiplier

        return self._last_state
    
    def _pop_calc(self):
        """批量计算，避免重复函数调用"""
        if len(self.value_stack) < 2:
            return
        
        b = self.value_stack.pop()
        a = self.value_stack.pop()
        op = self.op_stack.pop()

        if a is None or b is None:
            self.value_stack.append(None)
            return
        
        # 快速路径：处理常见情况
        if op == '+':
            self.value_stack.append(a + b)
        elif op == '-':
            self.value_stack.append(a - b)
        elif op == '*':
            self.value_stack.append(a * b)
        else:  # '/'
            if b == 0:
                self.value_stack.append(None)
            else:
                self.value_stack.append(a / b)
    
    def _try_calculate(self):
        """数字入栈后尝试计算"""
        while (len(self.value_stack) >= 2 and self.op_stack and 
               self.op_stack[-1] != '('):
            self._pop_calc()
    
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
    
    def get_full_trace(self) -> List[Tuple[float, int]]:
        """返回完整轨迹：[(step_id, partial_value, depth), ...]"""
        return [(t[0], t[1] if t[1] is not None else -999, t[2]) for t in self.trace]
    
    def get_masked_trace(self, mask_len: int) -> List[Tuple[float, int]]:
        """随机掩盖部分轨迹，增加鲁棒性"""
        full = self.get_full_trace()
        if len(full) <= 3:
            return full
        
        # 随机选择掩盖位置（类似BERT）
        mask_idx = random.randint(1, len(full)-2)
        masked = full[:mask_idx] + [(-1, -999, -1)] + full[mask_idx+1:]
        return masked
    
    def get_final_result(self) -> Optional[float]:
        """执行完成后获取最终结果"""
        if not self.trace:
            return None
        valid_traces = [t for t in self.trace if t[3] and t[1] is not None]
        return valid_traces[-1][1] if valid_traces else None


# 在游戏初始化后运行测试
if __name__ == "__main__":   
    game = Game()


