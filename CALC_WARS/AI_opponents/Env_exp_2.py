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
    

class Game:
    def __init__(self):
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
        self.state_vector = np.zeros(128, dtype=np.float32)
        self.info = {"message": ""}

        self.nonaction_times = 0
        self.noneffect_times = 0

        self.symbolic_executor = SymbolicExecutor()
        self.trainer = None  # 将由TrainingManager注入
        # 初始化游戏
        self.initial_deal()


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

        self.target = Target()
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
        self.state_vector = np.zeros(128, dtype=np.float32)
        self.leftright_unbal_bra = 0

        self.nonaction_times = 0
        self.noneffect_times = 0

        self.symbolic_executor.reset()
        # 重新发牌
        self.initial_deal()

        return self._get_state()

    
    def _get_state(self):
        """获取当前游戏状态的数值表示 - 优化版本"""
        # 预分配整个状态向量
        self.state_vector = np.zeros(128, dtype=np.float32)

        opponent = self.player2 if self.current_player == self.player1 else self.player1
        
        # 手牌信息
        # 当前玩家手牌 
        self.state_vector[:20] = np.array(self.current_player.hand[:20], dtype=np.float32) / 7.0
        
        # 对手手牌 
        self.state_vector[20:40] = np.array(opponent.hand[:20], dtype=np.float32) / 7.0
        
        # 技能牌信息
        # 当前玩家技能 
        self.state_vector[40:46] = np.array(self.current_player.skill_cards[:6], dtype=np.float32) / 5.0
        
        # 对手技能 
        self.state_vector[46:52] = np.array(opponent.skill_cards[:6], dtype=np.float32) / 5.0
        
        # 核心信息
        self.state_vector[52] = self.current_player.hp / 100.0
        self.state_vector[53] = opponent.hp / 100.0
        self.state_vector[54] = self.current_player.shield_count / 4.0
        self.state_vector[55] = opponent.shield_count / 4.0
        self.state_vector[56] = self.round_number / 8.0
        self.state_vector[57] = self.round_number ** 2 / 64.0
        self.state_vector[58] = self.step_count ** 1.5 / 5000.0
        self.state_vector[59] = self.continuous_operations / 5.0
        self.state_vector[60] = (self.state_vector[:20].sum() - self.state_vector[20:40].sum())/10.0 #手牌差距
        self.state_vector[61] = (self.current_player.hp - opponent.hp) / 100.0 #血量差距
        self.state_vector[62] = 1.0 if self.current_player == self.player1 else 0.0
        
        if self.current_player == self.player1:
            self.state_vector[63] = 1.0 if self.player1_round_end else 0.0
            self.state_vector[64] = 1.0 if self.player2_round_end else 0.0
        else:
            self.state_vector[63] = 1.0 if self.player2_round_end else 0.0
            self.state_vector[64] = 1.0 if self.player1_round_end else 0.0
    
        #  目标区域信息
        zones = [self.target.red_zone] +  self.target.yellow_zone[:2] +  self.target.blue_zone[:4] 
        self.state_vector[65:72] = np.array(zones[:7], dtype=np.float32) / 40.0 #归一化修改，放大注意

        if len(self.selected_cards) == 1:
            partial_result = self.selected_cards[0].value if self.selected_cards[0].card_type == CardType.NUMBER else None
        else:
            partial_result = self.calculate_expression(self.selected_cards)

        if partial_result is None:  #如果是None可能是AI选择了一个符号，比如3+，那就计算前面的部分
            partial_result = self.calculate_expression(self.selected_cards[:-1])
            op_last = True #不用管是否有效，函数里还会继续处理
        else:
            op_last = False

        self.state_vector[72:86] = self.reachable_check(op_last, partial_result).astype(np.float32)
        
        # 5. 选择状态 我们允许机器最大组合9位表达式
        selection_sequence = np.full((self.seq_len, 2), -1, dtype=np.float32)
        
        for idx, card in enumerate(self.selected_cards[:self.seq_len]):
            if card.card_type == CardType.NUMBER:
                # 维度0: 卡牌索引 (0-13)
                selection_sequence[idx, 0] = card.value
                # 维度1: 类型编码 (0=数字)
                selection_sequence[idx, 1] = 0.0
                
            elif card.card_type == CardType.OPERATOR:
                # 维度0: 运算符索引 (14-19)
                selection_sequence[idx, 0] = card.operator_type.value
                
                # 维度1: 运算符子类型
                if card.operator_type in [OperatorType.PLUS, OperatorType.MINUS]:
                    selection_sequence[idx, 1] = 1.0 # 加减
                elif card.operator_type in [OperatorType.MULTIPLY, OperatorType.DIVIDE]:
                    selection_sequence[idx, 1] = 2.0 # 乘除
                elif card.operator_type == OperatorType.LEFTBRA:
                    selection_sequence[idx, 1] = 3.0 # 左括号
                elif card.operator_type == OperatorType.RIGHTBRA:
                    selection_sequence[idx, 1] = 3.0 # 右括号（与左括号同类型）
        
        # 填充位置标记 (PAD)
        for idx in range(len(self.selected_cards), self.seq_len):
            selection_sequence[idx, 0] = 20.0 # PAD索引
            selection_sequence[idx, 1] = 4.0  # 空类型
        
        # 展平为18维向量
        self.state_vector[86:104] = selection_sequence.flatten() / 20.0

        # 5. 选择状态 高级特征
        expr_features = np.zeros(20, dtype=np.float32)

        if len(self.selected_cards) > 0:
            # 基础统计
            num_count = sum(1 for c in self.selected_cards if c.card_type == CardType.NUMBER)
            op_count = sum(1 for c in self.selected_cards if c.card_type == CardType.OPERATOR)
            
            expr_features[0] = len(self.selected_cards) / 9.0          # 总长度
            expr_features[1] = num_count / 6.0                        # 数字占比
            expr_features[2] = op_count / 5.0                         # 运算符占比
            
            # 括号平衡
            left_brackets = sum(1 for c in self.selected_cards 
                            if c.card_type == CardType.OPERATOR and c.operator_type == OperatorType.LEFTBRA)
            right_brackets = sum(1 for c in self.selected_cards 
                                if c.card_type == CardType.OPERATOR and c.operator_type == OperatorType.RIGHTBRA)
            expr_features[3] = left_brackets / 3.0                     # 左括号数
            expr_features[4] = right_brackets / 3.0                    # 右括号数
            expr_features[5] = (left_brackets - right_brackets)   # 括号不平衡度
            expr_features[6] = self.bracket_health()
            
            # 关键模式识别（引导AI）
            if len(self.selected_cards) >= 2:
                # 最后两张是否构成"运算符-数字"
                last_two = self.selected_cards[-2:]
                if (last_two[0].card_type == CardType.NUMBER and 
                    last_two[1].card_type == CardType.OPERATOR):
                    if last_two[1].operator_type != OperatorType.RIGHTBRA and \
                        last_two[1].operator_type != OperatorType.LEFTBRA:
                        expr_features[7] = 1.0
            
            # 价值预测（部分计算）
            if partial_result is not None:
                expr_features[8] = partial_result / 25.0  # 部分结果 归一化到25.0是为了注意
            if self.is_valid_expression(self.selected_cards):
                expr_features[9] = - 2.0  # 无效表达式标志

            # 新增：括号嵌套深度特征（AI能感知当前嵌套层级）
            max_depth = 0
            current_depth = 0
            for c in self.selected_cards:
                if c.card_type == CardType.OPERATOR:
                    if c.operator_type == OperatorType.LEFTBRA:
                        current_depth += 1
                        max_depth = max(max_depth, current_depth)
                    elif c.operator_type == OperatorType.RIGHTBRA:
                        current_depth -= 1
            expr_features[10] = max_depth / 3.0  # 最大嵌套深度（归一化）
            expr_features[11] = current_depth / 3.0  # 当前未闭合深度
            
            # 目标接近度
            target_zones = [self.target.red_zone] + self.target.yellow_zone + self.target.blue_zone
            if partial_result is not None:
                distances = [(partial_result - zone) for zone in target_zones]
                expr_features[12:19] = distances
            
        self.state_vector[104:124] = expr_features


        # 部分特征提取
        self.state_vector[124] = self.nonaction_times / 3.0 #无效动作统计
        self.state_vector[125] = (self.nonaction_times  ** 2) / 9.0
        total_actions = self.continuous_operations + self.nonaction_times + 1  # +1避免除零
        self.state_vector[126] = self.nonaction_times / total_actions


        # 游戏结束标志
        self.state_vector[127] = 1.0 if self.done else 0.0
        
        return self.state_vector
    
    def get_valid_actions_mask(self):
        # 动作空间设计：
        # 0-13: 选择手牌中的数字牌
        # 14-19: 选择手牌中的符号牌
        # 20-25: 使用技能牌
        # 26: 确认出牌（技能牌无需确认）
        # 27: 结束本回合
        # 28：结束本轮
        # 总共29种动作

        mask = np.zeros(29, dtype=np.int8)
    
        hand = np.array(self.current_player.hand[:20])
        skills = np.array(self.current_player.skill_cards[:6])
        
        # 向量化判断
        mask[:20] = (hand > 0).astype(np.int8)
        mask[20:26] = (skills > 0).astype(np.int8)

        # 2. 智能屏蔽：引导AI构造有效表达式
        if len(self.selected_cards) == 0:
            # 空序列：只允许数字或左括号
            mask[14:18] = 0  # 屏蔽+,-,*,/
            mask[19] = 0     # 屏蔽右括号
        else:
            last_card = self.selected_cards[-1]

            if len(self.selected_cards) >= 2:
            # 检查模式：[数字, 除法] → 禁止选择数字0
                second_last = self.selected_cards[-2]
                if (second_last.card_type == CardType.NUMBER and 
                    last_card.card_type == CardType.OPERATOR and 
                    last_card.operator_type == OperatorType.DIVIDE):
                    mask[0] = 0  # 禁用数字0
                    # 调试日志
                    if mask[0] == 1:
                        print(f"[MASK BUG] Division by zero protection failed!")

            if last_card.card_type == CardType.NUMBER:
                # 数字后：只允许运算符或右括号
                mask[:14] = 0  # 屏蔽数字
                mask[18] = 0   # 屏蔽左括号
            elif last_card.card_type == CardType.OPERATOR:
                if last_card.operator_type == OperatorType.LEFTBRA:
                    # (后：只允许数字或(
                    mask[14:20] = 0  # 屏蔽运算符
                elif last_card.operator_type == OperatorType.RIGHTBRA:
                    #右括号后不允许数字
                    mask[:14] = 0
                else:
                    # 运算符后：只允许数字或(
                    mask[14:18] = 0  # 屏蔽+,-,*,/
                    mask[19] = 0     # 屏蔽右括号

        leftright_unbal_bra = 0
        leftbra_count = sum(1 for card in self.selected_cards 
          if card.card_type == CardType.OPERATOR and card.operator_type == OperatorType.LEFTBRA)
        rightbra_count = sum(1 for card in self.selected_cards 
          if card.card_type == CardType.OPERATOR and card.operator_type == OperatorType.RIGHTBRA)
        leftright_unbal_bra = leftbra_count - rightbra_count
        #如果缺左括号匹配，不允许再加右括号
        if leftright_unbal_bra <= 0:
            mask[19] = 0

        #没有成对括号直接掩码
        left_bra = self.current_player.hand[18]
        right_bra = self.current_player.hand[19]
        if (left_bra + leftbra_count) == 0 or (right_bra + rightbra_count) == 0: #此处之前有重大bug，就是我仅仅统计了hand中的括号
            mask[18:20] = 0
                    
        # 条件设置
        mask[26] = 1 if len(self.selected_cards) >= 1 else 0
        #表达式长度限制
        if len(self.selected_cards) >= 9:
            mask[:26] = 0  # 太长，只允许确认动作

        mask[27] = 1 if not (self.player1_round_end or self.player2_round_end) else 0 
        mask[28] = 1 

        
        if self.nonaction_times >= 4:
            mask[:27] = 0 #无效多次结束

        if not mask.any():
            mask[27:] = 1  # 强制允许结束本轮
        
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
    
    def reachable_check(self, op_last = False, partial_result=None):
        """检查目标区域的可达性"""
        if partial_result is None:
            return np.full(14, -1, dtype=np.float32)
        
        reachable = np.full(14, -1, dtype=np.float32)
        for i, zone in enumerate(
            [self.target.red_zone] + self.target.yellow_zone + self.target.blue_zone
        ):
            for j in range(14):
                if self.current_player.hand[j]>0:
                    if op_last: #如果AI已经选好了符号，直接提示数字
                        last_operator = self.selected_cards[-1].operator_type
                        if last_operator == OperatorType.PLUS and abs(partial_result + j - zone) < 1e-3:
                            reachable[2*i] = 14 / 17.0
                            reachable[2*i+1] = j / 13.0
                        elif last_operator == OperatorType.MINUS and abs(partial_result - j - zone) < 1e-3:
                            reachable[2*i] = 15 / 17.0
                            reachable[2*i+1] = j / 13.0
                        elif last_operator == OperatorType.MULTIPLY and abs(partial_result * j - zone) < 1e-3:
                            reachable[2*i] = 16 / 17.0
                            reachable[2*i+1] = j / 13.0
                        elif last_operator == OperatorType.DIVIDE and j !=0 and abs(partial_result / j - zone) < 1e-3:
                            reachable[2*i] = 17 / 17.0
                            reachable[2*i+1] = j / 13.0
                    else:
                        if self.current_player.hand[17]>0 and j != 0 and abs(partial_result / j - zone) < 1e-3:
                            reachable[2*i] = 17 / 17.0
                            reachable[2*i+1] = j / 13.0
                        elif self.current_player.hand[14]>0 and abs(partial_result + j - zone) < 1e-3:
                            reachable[2*i] = 14 / 17.0
                            reachable[2*i+1] = j / 13.0
                        elif self.current_player.hand[15]>0 and abs(partial_result - j - zone) < 1e-3:
                            reachable[2*i] = 15 / 17.0
                            reachable[2*i+1] = j / 13.0
                        elif self.current_player.hand[16]>0 and abs(partial_result * j - zone) < 1e-3:
                            reachable[2*i] = 16 / 17.0
                            reachable[2*i+1] = j / 13.0
                else:
                    pass
        #除号用的最少所以有机会优先推荐除号
        #直接为AI推荐出牌路线
        return reachable
    
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

        time_penalty = -1.03 ** (self.step_count / 5.0) + 1 # 随时间递增
        
        self.brackert_count = 0
        self.info = {"message": ""}

        valid_mask = self.get_valid_actions_mask()

        # 掩码检查
        if valid_mask[action] == 0:
            reward = -40.0  # 标准非法动作惩罚
            reward += time_penalty
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

                    if not self.selected_cards:
                        reachable_tem = self.reachable_check(op_last=False, partial_result=action)
                        if sum(reachable_tem) == -14.0: #如果选出一张牌，什么区域也达不到，惩罚
                            reward += -5.0
                        else:
                            reward += 5.0

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
                    
                    self.leftright_unbal_bra = leftbra_count - rightbra_count

                    if self.leftright_unbal_bra > 0 and action == 19: #如果左括号匹配上右括号，大胜利
                        reward += 4.0

                reachable_zone = self.state_vector[72:86]
                if 3>= len(self.selected_cards) >=1:#如果选择推荐出牌路线，给奖励
                    for i, num in enumerate(reachable_zone):
                        if i%2 == 0 and abs(action / 17.0 - num) < 1e-3:
                            reward += 15.0      #符号正确
                            break
                        elif i%2 == 1 and abs(action / 17.0 - num) < 1e-3:
                            if self.selected_cards[-1].card_type == OperatorType(int((reachable_zone[i-1] + 1e-4)*17.0)):
                                reward += 80.0       #数字正确且符号也正确
                                break

                self.selected_cards.append(card)    #转换成表达式再把所有大于13的牌变为符号
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
                
                self.info["message"] = f"Used skill: {skill_type.name}"

                if skill_index == 0:
                    if self.current_player.hp <= 80:
                        reward += 10.0  # HEAL
                    elif self.current_player.hp <= 90:
                        reward += 5.0
                    else:
                        reward -= 10.0
                if skill_index == 1:
                    reward += 20.0  # STEAL
                if skill_index == 2:
                    reward += 15.0  # DRAW
                if skill_index == 3:
                    if self.current_player.shield_count <= 2:
                        reward += 10.0  #SHIELD
                    else:
                        reward -= 10.0
                if skill_index == 4:
                    reward += 15.0  # RUIN
                if skill_index == 5:
                    if opponent.shield_count >= 2:
                        reward += 10.0  # PIERCE
                    elif opponent.shield_count >=1:
                        reward += 7.0
                    else:
                        reward -= 10.0
                self.use_skill_card(skill_card)
            else:
                reward = -2.0 # 选择不存在的卡牌惩罚
                self.info["message"] = "Card not available"
            pass
        
        elif action == 27:  # 结束当前回合
            if self.continuous_operations == 0:
                if self.Operatability_check():
                    reward += -15.0
                else:
                    reward += -8.0
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
            reward += -15.0 * self.round_number

            if self.continuous_operations == 0:
                if self.Operatability_check():
                    reward += -15.0
                else:
                    reward += -8.0
            if self.selected_cards:
                reward += -10.0
            self.end_current_round()
            self.info["message"] = "Ended round"
        
        elif action == 26:  # 确认出牌
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
                    
                    reward += 0.0
                    # 奖励基于造成的伤害
                    if actual_damage == 50:
                        reward += 200.0 #伤害奖励
                    elif actual_damage == 30:
                        reward += 120.0
                    elif actual_damage == 10:
                        reward += 80.0

                    cards_used = len(self.selected_cards)
                    efficiency = actual_damage / max(cards_used, 1)
                    reward += efficiency * 2.0  # 牌效奖励
                
                    # 检查技能触发
                    skill_message = self.check_skill_triggers(result)
                    if skill_message:
                        reward += 2.0  # 技能触发奖励
                        self.info["skill_triggered"] = skill_message

                    if actual_damage == 0 and not skill_message:
                        self.noneffect_times += 1
                        reward += -10.0  # 如果不触发技能也不打伤害，惩罚浪费牌
                    else:
                        self.noneffect_times = 0

                    reward += 1.0 * self.brackert_count
    
                    self.selected_cards = []
                    self.symbolic_executor.reset()
                    self.continuous_operations += 1
                    self.info["message"] = f"Played cards, result: {result}, damage: {actual_damage}"
                else:
                    reward = -10.0 # 无效表达式惩罚
                    self.info["message"] = "Invalid expression"
                    #我突然意识到无效表达式不需要那么重的惩罚
                    #因为无效就返还卡牌，但是有效却不做事就是浪费牌
                    
                    # 返还选中的卡牌
                    for card in self.selected_cards:
                        self.current_player.add_card(card)
                    self.nonaction_times += 1
                    self.selected_cards = []
            else:
                reward = -10.0 # 卡牌数量不足惩罚
                # 这里的设计逻辑是：AI可能选了第一张牌就发现无法跟上组成技能或者伤害
                # 所以我们鼓励AI这个时候主动点confirm取消，也比浪费牌更好

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

        diff = max(-4, sum(self.current_player.hand[:20]) - sum(opponent.hand[:20])) * 0.5
        reward += diff

        # 添加连续操作奖励
        if self.continuous_operations > 0 and self.nonaction_times == 0:
            reward +=  1.0 * self.continuous_operations

        reward += - 5.0 * self.nonaction_times
        reward += - 0.5 * self.noneffect_times
        reward += time_penalty

        # 检查游戏是否结束
        if self.player1.hp <= 0 or self.player2.hp <= 0:
            self.done = True
            if self.current_player.hp > 0:
                reward = 1600.0 - 1.4 ** (self.step_count / 24.0) # 获胜奖励 鼓励速胜
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
            
            if result is not None and result >= 0:
                return int(result) if result == int(result) else round(result, 2)
            
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


