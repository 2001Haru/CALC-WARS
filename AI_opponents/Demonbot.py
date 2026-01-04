import numpy as np
from collections import deque
from Smart_solver import FastTemplateSolver

class DemonAgent:
    """
    基于决策树的专家级脚本机器人 (v2.1)
    策略核心：资源碾压 (Resource Overwhelming)
    修改日志：严格限制5张牌的使用，仅允许在斩杀或红区进攻时使用。
    """
    def __init__(self, logger=None):
        self.solver = FastTemplateSolver()
        self.logger = logger
        self.name = 'Demon_v1'
        # [核心] 内部动作缓存队列
        self.action_queue = deque()
        
        # 特殊数字映射表 (数字 -> Action Index Offset)
        self.special_num_map = {
            2:0, 6:1, 120:2, 8:3, 27:4, 64:5, 
            4:6, 9:7, 16:8, 25:9, 36:10, 49:11, 81:12, 100:13, 121:14, 144:15, 169:16, 
            24:17, 0:18, 1:19
        }
        
        # 技能动作索引
        self.skill_map = {
            'HEAL': 0, 'STEAL': 1, 'DRAW': 2, 'SHIELD': 3, 'RUIN': 4, 'PIERCE': 5
        }

        # 预定义的数字集合
        self.cubes = [8, 27, 64]
        self.squares = [4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169] 
        self.factorials = [2, 6, 120]

    def reset(self):
        self.action_queue.clear()

    def get_action(self, observation):
        if self.action_queue:
            return self.action_queue.popleft()

        state = self._parse_observation(observation)
        hand_counts = {i: c for i, c in enumerate(state['my_hand']) if c > 0}
        
        # r3: 3张牌能凑出的数字 (所有情况可用)
        # r5: 5张牌能凑出的数字 (仅斩杀或红区可用)
        r3, r5 = self.solver.get_reachable_sets(hand_counts)
        
        plan = self._evaluate_decision_tree(state, r3, r5)
        
        if not plan:
            end_action = 60 if state['op_ended_round'] else 61
            return end_action
            
        self.action_queue.extend(plan)
        return self.action_queue.popleft()

    def _parse_observation(self, obs):
        s = {}
        s['my_hand'] = np.round(obs[:20] * 5).astype(int)
        s['op_hand'] = np.round(obs[20:40] * 5).astype(int)
        s['my_skills'] = np.round(obs[40:46] * 3).astype(int)
        s['op_skills'] = np.round(obs[46:52] * 3).astype(int)
        s['my_hp'] = int(round(obs[52] * 120))
        s['op_hp'] = int(round(obs[53] * 120))
        s['my_shield'] = int(round(obs[54] * 4))
        s['op_shield'] = int(round(obs[55] * 4))
        
        zones_raw = np.round(obs[65:72] * 40).astype(int)
        s['zones'] = {
            'red': [zones_raw[0]], 
            'yellow': zones_raw[1:3].tolist(), 
            'blue': zones_raw[3:7].tolist() 
        }
        s['op_ended_round'] = (obs[64] > 0.5)
        s['my_hand_count'] = sum(s['my_hand'])
        s['op_hand_count'] = sum(s['op_hand'])
        return s

    def _evaluate_decision_tree(self, s, r3, r5):
        # 定义空集合，用于强制限制3张牌的场景
        r5_empty = set()

        # ==========================
        # 0. 基础本能反应
        # ==========================
        
        # 1. 斩杀检查 (例外：允许5张牌)
        if s['op_shield'] == 0:
            lethal_act = self._find_lethal(s, r3, r5) # 传入 r5 full
            if lethal_act: return [lethal_act]

        # 2. 紧急状态 (技能卡直接使用)
        if s['my_hp'] <= 100 and s['my_skills'][self.skill_map['HEAL']] > 0:
            return [54 + self.skill_map['HEAL']]
        if s['my_shield'] == 0 and s['my_skills'][self.skill_map['SHIELD']] > 0:
            return [54 + self.skill_map['SHIELD']]
        if s['my_skills'][self.skill_map['RUIN']] > 0 and s['op_hand_count'] >= 3:
            return [54 + self.skill_map['RUIN']]

        # ==========================
        # 复杂反应
        # ==========================
        diff = s['my_hand_count'] - s['op_hand_count']
        hp_diff = s['my_hp'] - s['op_hp']
        
        # 注意：以下 case 函数中，凡是涉及凑数/黄蓝区的，均传入 r5_empty
        # 凡是涉及红区的，单独处理或传入 r5 (full)
        
        if diff >= 0 and hp_diff >= 0:
            return self._case_1_advantage(s, r3, r5, r5_empty, diff)
        elif diff >= 0 and hp_diff >= -30:
            return self._case_2_slight_disadvantage(s, r3, r5, r5_empty, diff)
        elif diff >= 0 and hp_diff < -40:
            return self._case_3_heavy_defense(s, r3, r5, r5_empty, diff)
        else:
            return self._case_4_resource_deficit(s, r3, r5, r5_empty)

    # ------------------------------------------------------------------
    # 分支情况实现
    # ------------------------------------------------------------------

    def _case_1_advantage(self, s, r3, r5, r5_empty, diff):
        # A. 数量差 <= 6
        if diff <= 6:
            # 凑数严格限制3张 (r5_empty)
            combo = self._make_combo(['cube'], 'STEAL', r3, r5_empty)
            if combo: return combo
            
            # 蓝/黄区平方 (属于普通攻击，限制3张)
            target_squares = [x for x in self.squares if x in (s['zones']['blue'] + s['zones']['yellow'])]
            if target_squares:
                act = self._get_action_from_list(target_squares, r3, r5_empty, zone_priority=True, s=s)
                if act is not None: return [act, 54 + self.skill_map['DRAW']]
            
            combo = self._make_combo(['square'], 'DRAW', r3, r5_empty)
            if combo: return combo
            
            if s['op_shield'] > 0:
                combo = self._make_combo(['1'], 'PIERCE', r3, r5_empty)
                if combo: return combo
            else:
                # 进攻红区 (允许5张)
                act = self._get_action_from_list(s['zones']['red'], r3, r5, zone_priority=True, s=s)
                if act is not None: return [act]
                # 进攻黄区 (限制3张)
                act = self._get_action_from_list(s['zones']['yellow'], r3, r5_empty, zone_priority=True, s=s)
                if act is not None: return [act]
                
        # B. 数量差 > 6
        else:
            if s['op_shield'] > 0:
                # 蓝/黄区平方 (限制3张)
                target_squares = [x for x in self.squares if x in (s['zones']['blue'] + s['zones']['yellow'])]
                if target_squares:
                    act = self._get_action_from_list(target_squares, r3, r5_empty, zone_priority=True, s=s)
                    if act is not None: return [act, 54 + self.skill_map['DRAW']]
                
                if s['op_shield'] >= 2:
                    combo = self._make_combo(['1'], 'PIERCE', r3, r5_empty)
                    if combo: return combo
                    combo = self._make_combo(['square'], 'DRAW', r3, r5_empty)
                    if combo: return combo
                elif s['op_shield'] == 1:
                    combo = self._make_combo(['square'], 'DRAW', r3, r5_empty)
                    if combo: return combo
            
            else: # 无盾
                # 红区 (允许5张)
                act = self._get_action_from_list(s['zones']['red'], r3, r5, zone_priority=True, s=s)
                if act is not None: return [act]
                # 黄区 (限制3张)
                act = self._get_action_from_list(s['zones']['yellow'], r3, r5_empty, zone_priority=True, s=s)
                if act is not None: return [act]
                
                combo = self._make_combo(['square'], 'DRAW', r3, r5_empty)
                if combo: return combo
        
        return []

    def _case_2_slight_disadvantage(self, s, r3, r5, r5_empty, diff):
        # A. 数量差 <= 6
        if diff <= 6:
            combo = self._make_combo(['cube'], 'STEAL', r3, r5_empty)
            if combo: return combo
            
            target_squares = [x for x in self.squares if x in (s['zones']['blue'] + s['zones']['yellow'])]
            if target_squares:
                act = self._get_action_from_list(target_squares, r3, r5_empty, zone_priority=True, s=s)
                if act is not None: return [act, 54 + self.skill_map['DRAW']]
            combo = self._make_combo(['square'], 'DRAW', r3, r5_empty)
            if combo: return combo
            
            if s['op_shield'] > 0:
                combo = self._make_combo(['1'], 'PIERCE', r3, r5_empty)
                if combo: return combo
                act = self._make_num_action(['24'], r3, r5_empty)
                if act is not None: return [act]
            else:
                # 红区 (允许5张)
                act = self._get_action_from_list(s['zones']['red'], r3, r5, zone_priority=True, s=s)
                if act is not None: return [act]
                # 黄区 (限制3张)
                act = self._get_action_from_list(s['zones']['yellow'], r3, r5_empty, zone_priority=True, s=s)
                if act is not None: return [act]
                
                act = self._make_num_action(['24'], r3, r5_empty)
                if act is not None: return [act, 54 + self.skill_map['SHIELD']]

        # B. 数量差 > 6
        else:
            act = self._make_num_action(['24'], r3, r5_empty)
            if act is not None: return [act]
            
            act = self._make_num_action(['factorial'], r3, r5_empty)
            if act is not None: return [act]
            
            if s['op_shield'] > 0:
                target_squares = [x for x in self.squares if x in (s['zones']['blue'] + s['zones']['yellow'])]
                if target_squares:
                    act = self._get_action_from_list(target_squares, r3, r5_empty, zone_priority=True, s=s)
                    if act is not None: return [act, 54 + self.skill_map['DRAW']]
                
                if s['op_shield'] >= 2:
                    combo = self._make_combo(['1'], 'PIERCE', r3, r5_empty)
                    if combo: return combo
                    combo = self._make_combo(['square'], 'DRAW', r3, r5_empty)
                    if combo: return combo
                elif s['op_shield'] == 1:
                    combo = self._make_combo(['square'], 'DRAW', r3, r5_empty)
                    if combo: return combo
            else:
                # 红区 (允许5张)
                act = self._get_action_from_list(s['zones']['red'], r3, r5, zone_priority=True, s=s)
                if act is not None: return [act]
                # 黄区 (限制3张)
                act = self._get_action_from_list(s['zones']['yellow'], r3, r5_empty, zone_priority=True, s=s)
                if act is not None: return [act]

                combo = self._make_combo(['square'], 'DRAW', r3, r5_empty)
                if combo: return combo
                
        return []

    def _case_3_heavy_defense(self, s, r3, r5, r5_empty, diff):
        if diff <= 6:
            act = self._make_num_action(['24'], r3, r5_empty)
            if act is not None: return [act]
            act = self._make_num_action(['factorial'], r3, r5_empty)
            if act is not None: return [act]
            combo = self._make_combo(['cube'], 'STEAL', r3, r5_empty)
            if combo: return combo
            combo = self._make_combo(['square'], 'DRAW', r3, r5_empty)
            if combo: return combo
        else:
            return self._case_2_slight_disadvantage(s, r3, r5, r5_empty, diff=100)
        return []

    def _case_4_resource_deficit(self, s, r3, r5, r5_empty):
        combo = self._make_combo(['cube'], 'STEAL', r3, r5_empty)
        if combo: return combo
        
        combo = self._make_combo(['square'], 'DRAW', r3, r5_empty)
        if combo: return combo
        
        act = self._make_num_action(['24'], r3, r5_empty)
        if act is not None: return [act, 54 + self.skill_map['SHIELD'], 54 + self.skill_map['SHIELD']]
        
        return []

    # ------------------------------------------------------------------
    # 辅助功能
    # ------------------------------------------------------------------

    def _find_lethal(self, s, r3, r5):
        """寻找斩杀 (例外：任何区域都允许5张牌)"""
        op_hp = s['op_hp']
        targets = []
        if 50 >= op_hp: targets.extend(s['zones']['red'])
        if 30 >= op_hp: targets.extend(s['zones']['yellow'])
        if 10 >= op_hp: targets.extend(s['zones']['blue'])
        
        # 传入 Full R5
        return self._get_action_from_list(targets, r3, r5, zone_priority=True, s=s)

    def _make_combo(self, types, skill_name, r3, r5_set):
        """尝试组合: 造出数字 -> 使用技能"""
        act = self._make_num_action(types, r3, r5_set)
        if act is not None:
            return [act, 54 + self.skill_map[skill_name]]
        return None

    def _make_num_action(self, types, r3, r5_set):
        target_list = []
        if 'cube' in types: target_list.extend(self.cubes)
        if 'square' in types: target_list.extend(self.squares)
        if 'factorial' in types: target_list.extend(self.factorials)
        if '24' in types: target_list.append(24)
        if '1' in types: target_list.append(1)
        return self._get_action_from_list(target_list, r3, r5_set, zone_priority=False)

    def _get_action_from_list(self, targets, r3, r5_set, zone_priority=False, s=None):
        """
        在给定的目标列表中寻找可执行的动作
        注意：r5_set 可能是 Full Set (斩杀/红区) 或 Empty Set (普通情况)
        """
        for t in targets:
            eco_act = None
            std_act = None
            
            if zone_priority and s is not None:
                all_zones = s['zones']['red'] + s['zones']['yellow'] + s['zones']['blue']
                try:
                    idx = all_zones.index(t)
                    eco_act = 2 * idx
                    std_act = 2 * idx + 1
                except ValueError:
                    continue
            else:
                if t in self.special_num_map:
                    idx = self.special_num_map[t]
                    eco_act = 14 + 2 * idx
                    std_act = 14 + 2 * idx + 1
            
            # 1. 优先检查 3 张牌 (Efficiency)
            if eco_act is not None and t in r3:
                return eco_act
            # 2. 其次检查 5 张牌 (仅当 r5_set 包含 t 时)
            if std_act is not None and t in r5_set:
                return std_act
                
        return None