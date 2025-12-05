import pygame
import random
import math
from enum import Enum
from typing import List, Dict, Tuple, Optional
import os
import numpy as np
import torch
import torch.nn as nn
from CALC_WARS import Game
from AI_opponents.PPO_ver7_experiment_2 import SelfPlayAgent, PPOTransformer
from AI_opponents.Env_exp_2 import Card, CardType, OperatorType

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
# 创建AI实例

# PPO模型配置（需与训练时的配置一致）
PPO_CONFIG = {
    'obs_dim': 128,  # 必须与训练环境一致
    'action_dim': 29,
    'hidden_size': 512,  # Transformer 的 hidden_size
    'model_path': 'ppo_7000_2.pth',
    # Transformer 特有参数
    'seq_len': 9,
    'd_model': 128,
    'nhead': 4,
    'transformer_layers': 2
}

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

class SkillType_num(Enum):
    HEAL = 0          # 生命恢复牌
    STEAL = 1      # 盗窃牌
    DRAW = 2          # 抽牌
    SHIELD = 3      # 护盾牌
    RUIN = 4   #0牌
    PIERCE = 5 #1牌

Operator_num = {'+': 14,
            '-': 15,
            '*': 16,
            '/': 17,
            '(': 18,
            ')': 19}

# 技能牌字符串到数字的映射（用于get_card_index）
Skill_num_map = {
    "HE": 0,  # HEAL
    "ST": 1,  # STEAL
    "DR": 2,  # DRAW
    "SH": 3,  # SHIELD
    "RU": 4,  # RUIN
    "PI": 5   # PIERCE
}

class Card:
    def __init__(self, value: str, card_type: CardType, skill_type: Optional[SkillType] = None):
        self.value = value
        self.card_type = card_type
        self.skill_type = skill_type
        self.used = False
    
    def __str__(self):
        return self.value

class Player:
    def __init__(self, name: str):
        self.name = name
        self.hp = 100
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
        self.yellow_zone = random.sample(range(24, 37),2)
        self.blue_zone = random.sample([1,2,3,5,6,7,8,10,11,12,13,14,15,17,18,19,20,21,22,23,24],3)+ random.sample([4,9,16],1)
    
    def get_damage(self, result: int) -> int:
        if result == self.red_zone:
            return 50
        elif result in self.yellow_zone:
            return 30
        elif result in self.blue_zone:
            return 10
        return 0
    
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
        text_surface = game.font.render(self.text, True, self.text_color)
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

    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("CALC WARS")
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
        self.selected_cards = []
        self.calculation_result = None
        self.message = ""
        self.history = []  # 历史记录
        self.action_messages = []  # 操作消息队列
        self.player1_round_end = False  # 玩家1是否结束本轮
        self.player2_round_end = False  # 玩家2是否结束本轮
        self.continuous_operations = 0  # 连续操作次数
        self.first_to_end_round = None  # 当前轮次先结束本轮的玩家
        self.nonaction_times = 0
        self.step_count = 0  # 总步数统计
        self.seq_len = 9
        self.done = False  # 游戏是否结束

        self.calculation_display_time = 0  # 记录结果显示时间
        self.result_alpha = 255  # 透明度(255为完全不透明)
        self.showing_result = False  # 是否正在显示结果
        self.fade_duration = 1500  # 淡出持续时间(毫秒)
        self.broadcastmessage = ""
        # 广播消息显示相关
        self.broadcast_display_time = 0  # 广播消息显示时间
        self.broadcast_alpha = 255  # 广播消息透明度
        self.showing_broadcast = False  # 是否正在显示广播消息

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
            "Square: One Draw Card( Draw 3 cards randomly )",
            "24 Points: Two Shield Cards( Get 1 shield )",
            "ZERO: One Ruin card( Destroy 3 cards ramdomly )",
            "ONE: One Pierce card( Break the opponent's shield )",
            '',
            'If the difference in hand cards between the two players is 6 or more',
            'Stealing only steals 1 card!!'
        ]

        # === PPO AI 初始化 ===
        self._init_ppo_ai()

        # 初始化游戏
        self.initial_deal()

    def _init_ppo_ai(self):
        """初始化PPO AI - 抽取成独立方法"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n=== Initializing PPO AI on {self.device} ===")
        
        # 初始化 Transformer 网络
        self.ppo_policy = PPOTransformer(
            obs_dim=PPO_CONFIG['obs_dim'],
            action_dim=PPO_CONFIG['action_dim'],
            hidden_size=PPO_CONFIG['hidden_size'],
            seq_len=PPO_CONFIG['seq_len'],
            d_model=PPO_CONFIG['d_model'],
            nhead=PPO_CONFIG['nhead'],
            num_layers=PPO_CONFIG['transformer_layers']
        ).to(self.device)
        
        # 加载训练好的权重
        try:
            checkpoint = torch.load(PPO_CONFIG['model_path'], 
                                map_location=self.device, 
                                weights_only=False)
            self.ppo_policy.load_state_dict(checkpoint['policy_net_state_dict'])
            self.ppo_policy.eval()
            print(f"✓ Loaded checkpoint from episode {checkpoint.get('episode', 'unknown')}")
            print(f"✓ Model file: {PPO_CONFIG['model_path']}")
        except FileNotFoundError:
            print(f"✗ Warning: Model file '{PPO_CONFIG['model_path']}' not found. Using random weights!")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
        
        # 创建智能体
        self.ppo_agent = SelfPlayAgent(self.ppo_policy, self.device)
        print("PPO AI initialized successfully\n")


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
        # 按优先级尝试不同的字体名称
        font_names = [
            "times new roman",  # Windows/Linux常见名称
            "Times New Roman",  # 首字母大写
            "Times",            # macOS系统Times字体
            "times",            # 全小写
            None                # 系统默认字体
        ]
        
        for name in font_names:
            try:
                return pygame.font.SysFont(name, size)
            except:
                continue
        
        # 如果所有尝试都失败，使用pygame的默认字体
        return pygame.font.Font(None, size)
    
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
        for _ in range(7):
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
            return Card(str(random.randint(0, 13)), CardType.NUMBER)
        else:  # 33.333%概率符号牌
            operators = ['+','+','-','-', '*', '*','/','+','-','*','+','+','-','-', '*', '*','/','/','/','+','-','*','(', ')'] 
            #加号概率：减号概率：乘号概率：除号概率：括号概率=6:6:6:4:2
            return Card(random.choice(operators), CardType.OPERATOR)
    
    def generate_skill_card(self, skill_type: SkillType) -> Card:
        """生成技能牌"""
        return Card(skill_type.value, CardType.SKILL, skill_type)
    
    def calculate_expression(self, cards: List[Card]) -> Optional[int]:
        """计算表达式结果"""
        try:
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
            self.current_player.hp = min(100, self.current_player.hp + 20)
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
                for i in range(3):
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
            for _ in range(3):
                self.current_player.add_card(self.generate_random_card())
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
        self.selected_cards = []
        self.nonaction_times = 0
        self.current_player.is_active = False
        
        self.current_player = self.player2 if self.current_player == self.player1 else self.player1
        self.current_player.is_active = True
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
        self.nonaction_times = 0
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
        
        # 清空历史消息和操作消息
        self.history = []
        self.action_messages = []
        
        self.message = f"Round {self.round_number} starts, {self.current_player.name} goes first"
    
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
                    makeupcard = self.generate_random_card()
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
        
        '''self.screen.blit(self.font.render(player1_text, True, GOLD_WARM), (20, 10))
        self.screen.blit(self.font.render(player2_text, True, GOLD_WARM), (20, 40))'''
        self.screen.blit(self.font.render(player1_text, True, GOLD_CLASSIC), (65, SCREEN_HEIGHT - 340))
        self.screen.blit(self.font.render(player2_text, True, GOLD_CLASSIC), (695, SCREEN_HEIGHT - 340))

        #绘制血槽
        health_bar_width = 200
        health_bar_height = 20
        current_health_width1 = (self.player1.hp / 100) * health_bar_width
        pygame.draw.rect(self.screen, YELLOW_CREAM, (65, SCREEN_HEIGHT - 360, health_bar_width, health_bar_height))
        pygame.draw.rect(self.screen, RED_ELECTRIC, (65, SCREEN_HEIGHT - 360, current_health_width1, health_bar_height))
        #绘制边框
        pygame.draw.rect(self.screen, BLACK, (65, SCREEN_HEIGHT - 360, health_bar_width, health_bar_height), 2)

        current_health_width2 = (self.player2.hp / 100) * health_bar_width
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
        '''target_text = f"Target - Red: {self.target.red_zone} Yellow: {self.target.yellow_zone} Blue: {self.target.blue_zone}"
        self.screen.blit(self.font.render(target_text, True, WHITE_CREAM), (500, 10))'''
        
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

    def get_card_index(self, card: Card):
        """获取卡牌在vec中的索引"""
        if card.card_type == CardType.NUMBER:
            return int(card.value)
        elif card.card_type == CardType.OPERATOR:
            # 正确：使用Operator_num字典
            return Operator_num[card.value]
        elif card.card_type == CardType.SKILL:
            # 正确：使用Skill_num_map字典
            return Skill_num_map[card.value]
        
    def hand_index_get_card(self,card_num:int):
        if 0<= card_num <= 13:
            return Card(value = str(card_num), card_type = CardType.NUMBER)
        if 14<= card_num <=19:
            op_map = {14:'+', 15:'-', 16:'*', 17:'/', 18:'(', 19:')'}
            return Card(value = op_map[card_num], card_type = CardType.OPERATOR)
        
    def skill_index_get_card(self,card_num:int):
        skill_map = {0:SkillType.HEAL, 1:SkillType.STEAL, 2:SkillType.DRAW, 3:SkillType.SHIELD, 4:SkillType.RUIN, 5:SkillType.PIERCE}
        return self.generate_skill_card(skill_map[card_num])
    
    # 在 Game 类中添加
    def update_vec(self):
        """
        重新计算 vec_hand 和 vec_skill，确保与 hand/skill_cards 一致
        AI回合会额外扣除 selected_cards 中的牌（未确认的选择）
        """
        # 1. 重新计算当前玩家
        self.current_player.vec_hand = [0] * 20
        self.current_player.vec_skill = [0] * 6
        
        # 统计手牌
        for card in self.current_player.hand:
            if card.card_type == CardType.NUMBER:
                self.current_player.vec_hand[int(card.value)] += 1
            elif card.card_type == CardType.OPERATOR:
                self.current_player.vec_hand[Operator_num[card.value]] += 1
        
        # 统计技能牌
        for card in self.current_player.skill_cards:
            skill_num = Skill_num_map[card.skill_type.value]
            self.current_player.vec_skill[skill_num] += 1
        
        # 关键：扣除已选择但未确认的卡牌
        for card in self.selected_cards:
            if card.card_type != CardType.SKILL:  # 只处理手牌
                idx = self.get_card_index(card)
                if 0 <= idx < 20 and self.current_player.vec_hand[idx] > 0:
                    self.current_player.vec_hand[idx] -= 1
        
        # 2. 重新计算对手
        opponent = self.player2 if self.current_player == self.player1 else self.player1
        opponent.vec_hand = [0] * 20
        opponent.vec_skill = [0] * 6
        
        for card in opponent.hand:
            if card.card_type == CardType.NUMBER:
                opponent.vec_hand[int(card.value)] += 1
            elif card.card_type == CardType.OPERATOR:
                opponent.vec_hand[Operator_num[card.value]] += 1
        
        for card in opponent.skill_cards:
            skill_num = Skill_num_map[card.skill_type.value]
            opponent.vec_skill[skill_num] += 1

    
    def get_state(self):
        """生成121维状态向量 - 纯计算，不修改游戏逻辑"""
        # 1. 强制同步vec状态（仅用于AI观察，不影响真实手牌）
        self.update_vec()
        
        state_vector = np.zeros(128, dtype=np.float32)
        opponent = self.player2 if self.current_player == self.player1 else self.player1
        
        # 区域1-2：手牌信息（0-51维）
        state_vector[:20] = np.array(self.current_player.vec_hand[:20]) / 7.0
        state_vector[20:40] = np.array(opponent.vec_hand[:20]) / 7.0
        state_vector[40:46] = np.array(self.current_player.vec_skill[:6]) / 5.0
        state_vector[46:52] = np.array(opponent.vec_skill[:6]) / 5.0
        
        # 区域3：核心信息（52-61维）
        state_vector[52] = self.current_player.hp / 100.0
        state_vector[53] = opponent.hp / 100.0
        state_vector[54] = self.current_player.shield_count / 6.0
        state_vector[55] = opponent.shield_count / 6.0
        state_vector[56] = self.round_number / 5.0  # 训练环境使用/5.0
        state_vector[57] = (self.round_number ** 2) / 25.0  # 新增：平方特征
        state_vector[58] = self.step_count ** 1.5 / 5000.0
        state_vector[59] = self.continuous_operations / 5.0
        state_vector[60] = (self.current_player.hp - opponent.hp) / 100.0
        state_vector[61] = (state_vector[:20].sum() - state_vector[20:40].sum()) / 10.0
        state_vector[62] = 1.0 if self.current_player == self.player1 else 0.0
        
        if self.current_player == self.player1:
            state_vector[63] = 1.0 if self.player1_round_end else 0.0
            state_vector[64] = 1.0 if self.player2_round_end else 0.0
        else:
            state_vector[63] = 1.0 if self.player2_round_end else 0.0
            state_vector[64] = 1.0 if self.player1_round_end else 0.0
        
        # 区域5：目标区域（66-72维）
        zones = [self.target.red_zone] + self.target.yellow_zone[:2] + self.target.blue_zone[:4]
        state_vector[65:72] = np.array(zones[:7]) / 60.0
    
        if len(self.selected_cards) == 1:
            partial_result = int(self.selected_cards[0].value) if self.selected_cards[0].card_type == CardType.NUMBER else None
        else:
            partial_result = self.calculate_expression(self.selected_cards)

        if partial_result is None:  #如果是None可能是AI选择了一个符号，比如3+，那就计算前面的部分
            partial_result = self.calculate_expression(self.selected_cards[:-1])
            op_last = True #不用管是否有效，函数里还会继续处理
        else:
            op_last = False

        state_vector[72:86] = self.reachable_check(op_last=op_last,partial_result=partial_result).astype(np.float32)
        
        selection_sequence = np.full((self.seq_len, 2), -1, dtype=np.float32)
        
        for idx, card in enumerate(self.selected_cards[:self.seq_len]):
            if card.card_type == CardType.NUMBER:
                # 维度0: 卡牌索引 (0-13)
                selection_sequence[idx, 0] = card.value
                # 维度1: 类型编码 (0=数字)
                selection_sequence[idx, 1] = 0.0
                
            elif card.card_type == CardType.OPERATOR:
                # 维度0: 运算符索引 (14-19)
                selection_sequence[idx, 0] = self.get_card_index(card)
                
                # 维度1: 运算符子类型
                if card.value in ['+','-']:
                    selection_sequence[idx, 1] = 1.0  # 加减
                elif card.value in ['*','/']:
                    selection_sequence[idx, 1] = 2.0  # 乘除
                elif card.value == '(':
                    selection_sequence[idx, 1] = 3.0  # 左括号
                elif card.value == ')':
                    selection_sequence[idx, 1] = 3.0  # 右括号（与左括号同类型）

        # 填充位置标记 (PAD)
        for idx in range(len(self.selected_cards), self.seq_len):
            selection_sequence[idx, 0] = 20.0  # PAD索引
            selection_sequence[idx, 1] = 4.0   # 空类型

        state_vector[86:104] = selection_sequence.flatten() / 20.0
        
        # 区域7：高级特征
        expr_features = np.zeros(20, dtype=np.float32)
        if len(self.selected_cards) > 0:
            num_count = sum(1 for c in self.selected_cards if c.card_type == CardType.NUMBER)
            op_count = sum(1 for c in self.selected_cards if c.card_type == CardType.OPERATOR)
            
            expr_features[0] = len(self.selected_cards) / 9.0
            expr_features[1] = num_count / 6.0
            expr_features[2] = op_count / 5.0
            
            # 括号统计
            left_brackets = sum(1 for c in self.selected_cards if c.value == '(')
            right_brackets = sum(1 for c in self.selected_cards if c.value == ')')
            expr_features[3] = left_brackets / 3.0
            expr_features[4] = right_brackets / 3.0
            expr_features[5] = (left_brackets - right_brackets)  # 不平衡度
            expr_features[6] = self.bracket_health()
            
            # 模式识别
            if len(self.selected_cards) >= 2:
                last_two = self.selected_cards[-2:]
                if (last_two[0].card_type == CardType.NUMBER and 
                    last_two[1].card_type == CardType.OPERATOR):
                    if last_two[1].value != ')' and last_two[1].value != ')':
                        expr_features[7] = 1.0
            
            # 价值预测
            if partial_result is not None:
                expr_features[8] = partial_result / 100.0  # 部分结果
                if partial_result < 0:
                    expr_features[9] = - 5.0  # 无效表达式标志
            
            # 括号深度
            max_depth = current_depth = 0
            for c in self.selected_cards:
                if c.value == '(':
                    current_depth += 1
                    max_depth = max(max_depth, current_depth)
                elif c.value == ')':
                    current_depth -= 1
            expr_features[10] = max_depth / 3.0
            expr_features[11] = current_depth / 3.0
            
            # 目标接近度
            target_zones = [self.target.red_zone] + self.target.yellow_zone + self.target.blue_zone
            if partial_result is not None:
                distances = [(partial_result - zone) for zone in target_zones]
                expr_features[12:19] = distances

        state_vector[104:124] = expr_features
        
        # 区域8：无效动作统计
        state_vector[124] = self.nonaction_times / 3.0
        state_vector[125] = (self.nonaction_times ** 2) / 9.0
        total_actions = self.continuous_operations + self.nonaction_times + 1  # +1避免除零
        state_vector[126] = self.nonaction_times / total_actions

        
        # 区域9：游戏结束标志（120维）
        state_vector[127] = 1.0 if self.done else 0.0
        
        return state_vector
    
    
    def get_mask(self):
        """生成29维动作掩码 - 不修改游戏逻辑，只计算掩码"""
        self.update_vec()
        mask = np.zeros(29, dtype=np.int8)
        
        # 基础可用性
        hand_counts = np.array(self.current_player.vec_hand[:20], dtype=np.int8)
        skill_counts = np.array(self.current_player.vec_skill[:6], dtype=np.int8)
        mask[:20] = (hand_counts > 0).astype(np.int8)
        mask[20:26] = (skill_counts > 0).astype(np.int8)
        
        # 表达式规则
        if len(self.selected_cards) == 0:
            mask[14:18] = 0  # 屏蔽运算符
            mask[19] = 0     # 屏蔽右括号
        else:
            last_card = self.selected_cards[-1]
            
            # 除零保护
            if len(self.selected_cards) >= 2:
                second_last = self.selected_cards[-2]
                if (second_last.card_type == CardType.NUMBER and 
                    last_card.card_type == CardType.OPERATOR and 
                    last_card.value == '/'):
                    mask[0] = 0
            
            if last_card.card_type == CardType.NUMBER:
                mask[:14] = 0
                mask[18] = 0
            elif last_card.card_type == CardType.OPERATOR:
                if last_card.value == '(':
                    mask[14:20] = 0
                elif last_card.value == ')':
                    mask[:14] = 0
                else:
                    mask[14:18] = 0
                    mask[19] = 0
        
        # 括号平衡
        leftbra_count = sum(1 for c in self.selected_cards if c.value == '(')
        rightbra_count = sum(1 for c in self.selected_cards if c.value == ')')
        if leftbra_count <= rightbra_count:
            mask[19] = 0

        #没有成对括号直接掩码
        left_bra = self.current_player.vec_hand[18]
        right_bra = self.current_player.vec_hand[19]
        if (left_bra + leftbra_count) == 0 or (right_bra + rightbra_count) == 0: #此处之前有重大bug，就是我仅仅统计了hand中的括号
            mask[18:20] = 0
        
        # 确认按钮
        '''mask[26] = 1 if len(self.selected_cards) >= 3 else 0'''
        mask[26] = 1 if len(self.selected_cards) >= 1 else 0

        if self.nonaction_times >= 4:
            mask[:27] = 0
        
        # 长度限制
        if len(self.selected_cards) >= 9:
            mask[:26] = 0
        
        # 结束回合
        mask[27] = 1 if (not self.selected_cards and 
                        not (self.player1_round_end or self.player2_round_end)) else 0
        # 结束本轮
        mask[28] = 1
    
        if not mask.any():
            mask[27:] = 1  # 强制允许结束本轮
        
        return mask
    
    def Operatability_check(self):
        """检查是否可以进行有效操作 - 与Game_env_PPO.py保持一致"""
        ava_op = 0
        ava_num = 0
        
        # 统计手牌中的数字牌（0-13）
        for card in self.current_player.hand:
            if card.card_type == CardType.NUMBER:
                ava_num += 1
        
        # 统计手牌中的运算符牌（+,-,*,/）
        for card in self.current_player.hand:
            if card.card_type == CardType.OPERATOR and card.value in ['+', '-', '*', '/']:
                ava_op += 1
        
        # 统计已选中的卡牌
        for card in self.selected_cards:
            if card.card_type == CardType.NUMBER:
                ava_num += 1
            elif card.card_type == CardType.OPERATOR and card.value in ['+', '-', '*', '/']:
                ava_op += 1

        return ava_num >= 2 and ava_op >= 1
            
    def reachable_check(self, op_last = False, partial_result=None):
        """检查目标区域的可达性"""
        if partial_result is None:
            return np.full(14, -1, dtype=np.float32)
        
        self.update_vec()
        
        reachable = np.full(14, -1, dtype=np.float32)
        for i, zone in enumerate(
            [self.target.red_zone] + self.target.yellow_zone + self.target.blue_zone
        ):
            for j in range(14):
                if self.current_player.vec_hand[j]>0:
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
                        if self.current_player.vec_hand[17]>0 and j != 0 and abs(partial_result / j - zone) < 1e-6:
                            reachable[2*i] = 17 / 17.0
                            reachable[2*i+1] = j / 13.0
                        elif self.current_player.vec_hand[14]>0 and abs(partial_result + j - zone) < 1e-6:
                            reachable[2*i] = 14 / 17.0
                            reachable[2*i+1] = j / 13.0
                        elif self.current_player.vec_hand[15]>0 and abs(partial_result - j - zone) < 1e-6:
                            reachable[2*i] = 15 / 17.0
                            reachable[2*i+1] = j / 13.0
                        elif self.current_player.vec_hand[16]>0 and abs(partial_result * j - zone) < 1e-6:
                            reachable[2*i] = 16 / 17.0
                            reachable[2*i+1] = j / 13.0
        #除号用的最少所以有机会优先推荐除号
        #直接为AI推荐出牌路线
        return reachable

    def bracket_health(self):
        """验证卡牌健康度 - 与Game_env_PPO.py保持一致"""
        if len(self.selected_cards) < 3:
            return 0.7

        health = 0.0
        left_stack = []  # 存储未匹配的左括号位置
        illegal_rights = 0  # 非法右括号计数

        for i, c in enumerate(self.selected_cards):
            if c.card_type == CardType.OPERATOR:
                if c.value == '(':
                    left_stack.append(i)  # 压栈
                
                elif c.value == ')':
                    if left_stack:
                        left_stack.pop()  # 成功匹配，出栈
                    else:
                        illegal_rights += 1  # 灾难：右括号无左匹配
        
        unmatched_lefts = len(left_stack)  # 轻微错误：左括号未关闭
        
        health += unmatched_lefts * 0.2 + 0.75 * illegal_rights
        return health
    
    def act_action(self, action):
        # 确保这是AI回合
        if self.current_player != self.player2:
            print("Warning: AI action called on human player's turn!")
            return

        self.update_vec()
        self.step_count += 1
        opponent = self.player1
            
        # 解析动作
        if action < 20 :  # 选择手牌
            if self.current_player.vec_hand[action] > 0:
                # 遍历找到手牌中对应的**真实卡牌对象**
                found_card = None
                for card in self.current_player.hand:
                    card_idx = self.get_card_index(card)
                    if card_idx == action:
                        found_card = card
                        break
                
                if found_card:
                    self.selected_cards.append(found_card)
                    print(f"AI selected card: {found_card}")
                else:
                    print(f"Warning: Could not find card with index {action} in hand")
            else:
                print(f"Warning: Card index {action} not available")

        elif action < 26 :  # 使用技能牌
            skill_index = action - 20
            if self.current_player.vec_skill[skill_index] > 0:
                # 遍历找到技能牌中对应的**真实卡牌对象**
                found_skill = None
                for card in self.current_player.skill_cards:
                    card_idx = self.get_card_index(card)
                    if card_idx == skill_index:
                        found_skill = card
                        break
                
                if found_skill:
                    self.use_skill_card(found_skill)
                    print(f"AI used skill card: {found_skill}")
                else:
                    print(f"Warning: Could not find skill card with index {action}")
            else:
                print(f"Warning: Skill index {action-20} not available")
                
        elif action == 27:  # 结束当前回合
            if not self.player1_round_end and not self.player2_round_end:
                # 补牌
                makeuplist = []
                for _ in range(2):
                    makeupcard = self.generate_random_card()
                    self.current_player.add_card(makeupcard)
                    makeuplist.append(makeupcard.value)
                self.broadcastmessage = f"{self.current_player.name} ended continuous operations and drew {' '.join(makeuplist)}"
                self.broadcast_display_time = pygame.time.get_ticks()
                self.broadcast_alpha = 255
                self.showing_broadcast = True
            self.switch_player()
            
        elif action == 28:  # 结束本轮
            self.end_current_round()
            
        elif action == 26:  # 确认出牌
            if len(self.selected_cards) >= 3:
                result = self.calculate_expression(self.selected_cards)
                if result is not None and result >= 0:
                    self.calculation_result = result
                    damage = self.target.get_damage(result)
                    actual_damage = opponent.take_damage(damage)
                        
                        # 检查技能触发
                    skill_triggered = self.check_skill_triggers(result)

                    self.calculation_display_time = pygame.time.get_ticks()  # 记录开始显示时间
                    self.result_alpha = 255  # 重置为完全不透明
                    self.showing_result = True

                    # 添加操作消息
                    if actual_damage > 0:
                        self.play_sound_effect('damage')
                        self.add_action_message(f"{self.current_player.name} calculated {result}, causing {opponent.name} {actual_damage} points of damage!")
                        self.broadcastmessage = f"{self.current_player.name} calculated {result} and caused {opponent.name} {actual_damage} points of damage!"
                    
                     # 添加广播消息
                    self.broadcast_display_time = pygame.time.get_ticks()
                    self.broadcast_alpha = 255
                    self.showing_broadcast = True
                            
                    
                    self.message = f"Result: {result}, Damage: {actual_damage}"
                    if damage > 0 and actual_damage == 0:
                        self.message += f" ({damage} damage blocked by shield)"
                        self.play_sound_effect('shield_block')

                    # 关键修复：安全移除卡牌（基于内容匹配而非对象身份）
                    for selected_card in self.selected_cards:
                        # 遍历查找匹配的卡牌对象
                        for hand_card in self.current_player.hand[:]:  # 使用副本可安全删除
                            if (hand_card.value == selected_card.value and 
                                hand_card.card_type == selected_card.card_type):
                                self.current_player.hand.remove(hand_card)
                                break
                        else:
                            print(f"Warning: Could not remove {selected_card}, not found in hand")
                            
                    self.nonaction_times = 0
                    self.selected_cards = []
                    self.continuous_operations += 1
                else:
                    self.message = "Invalid expression"
                    self.selected_cards = []
                    self.nonaction_times += 1
            else:
                self.message = "Need at least 3 cards"
                self.selected_cards = []
                self.nonaction_times += 1


    def get_ai_action(self):
        """获取PPO AI的动作"""
        if not hasattr(self, 'ppo_agent'):
            print("Error: PPO AI not initialized!")
            return 0
        
        # 获取当前状态（当前玩家视角）
        state = self.get_state()
        
        # 获取动作掩码
        action_mask = self.get_mask()
        
        # 确保AI视角是player2
        if self.current_player != self.player2:
            print("Warning: AI is being called for non-AI player")
            return 0
        
        # 关键：使用no_grad上下文
        with torch.no_grad():
            action, log_prob, value = self.ppo_agent.get_action(
                state, action_mask, player_id=1, deterministic=True
            )
        
        return action

                
    def run(self):
        """运行游戏"""
        #self.play_background_music()  # 开始播放背景音乐
        # 开始播放第一首音乐
        self.play_next_music()
        running = True
    
        while running:
            mouse_pos = pygame.mouse.get_pos()
            mouse_pressed = pygame.mouse.get_pressed()  # 获取当前鼠标状态
    
            # PPO AI回合
            if self.current_player == self.player2 and not self.done:
                self.update_vec()
                print(self.current_player.vec_hand)
                print(self.current_player.vec_skill)
                print(self.Operatability_check())

                state = self.get_state()
                mask = self.get_mask()

                # AI 思考（可调整确定性）
                with torch.no_grad():
                    action, log_prob, value = self.ppo_agent.get_action(
                        state, mask, player_id=1, deterministic=True  # 对战时用确定性策略
                    )

                print(f"[AI Turn] Action: {action} | LogProb: {log_prob:.3f} | Value: {value:.3f}")

                self.act_action(action)
                pygame.time.wait(200)  # 给AI操作添加短暂延迟，便于观察

                self.update_vec()
                    
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
                self.done = True
            elif self.player2.hp <= 0:
                self.message = "Player 1 Wins"
                win_font = pygame.font.SysFont('Arial', 240, bold=True)
                win = win_font.render("Player1 WINS!", True, ORANGE_PEACH)  # 深蓝色标题
                self.play_sound_effect('win')
                self.screen.blit(win, (0, 300))
                pygame.display.flip()  # 确保消息显示
                pygame.time.wait(5000)
                running = False
                self.done = True
        
        pygame.quit()

game = Game()

if __name__ == "__main__":
    game.run()
