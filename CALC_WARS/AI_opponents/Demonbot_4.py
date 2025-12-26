import numpy as np
from collections import deque
from Smart_solver import FastTemplateSolver

class DemonAgent_V4:
    """
    四代恶魔机器人 (The Guardian / Counter-Attacker)
    特点：
    奇异技术
    """
    def __init__(self, logger=None):
        self.solver = FastTemplateSolver()
        self.logger = logger
        self.name = 'Demon_v5'
        self.action_queue = deque()
        
        self.special_num_map = {
            2:0, 6:1, 120:2, 8:3, 27:4, 64:5, 
            4:6, 9:7, 16:8, 25:9, 36:10, 49:11, 81:12, 100:13, 121:14, 144:15, 169:16, 
            24:17, 0:18, 1:19
        }
        self.skill_type_map = {
            'HE': 0, 'ST': 1, 'DR': 2, 'SH': 3, 'RUIN': 4, 'PI': 5
        }

    def reset(self):
        self.action_queue.clear()

    def get_action(self, observation):
        if self.action_queue:
            return self.action_queue.popleft()

        state = self._parse_observation(observation)
        
        # 决策
        plan = self._evaluate_decision_tree(state)
        
        if not plan:
            plan = [61] if state['op_ended_round'] else [60] 
            
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
        return s

    def _evaluate_decision_tree(self, state):
        hand_counts = {i: c for i, c in enumerate(state['my_hand']) if c > 0}
        r3, r5_real = self.solver.get_reachable_sets(hand_counts)
        
        # 三代恶魔在防守时非常谨慎，通常只用3张牌，除非是为了保命(斩杀对手)
        r5_empty = set() 

        # ==========================
        # 1. 基础本能 (Instinct)
        # ==========================
        
        # [斩杀]
        lethal_act = self._find_lethal(state, r3, r5_real)
        if lethal_act: return [lethal_act]

        # [HEAL] 血量 <= 100 (几乎总是回血)
        if state['my_hp'] <= 100 and state['my_skills'][0] > 0: return [54 + 0] 
        
        # [SHIELD] 无盾必补
        if state['my_shield'] == 0 and state['my_skills'][3] > 0: return [54 + 3]
        
        # [RUIN] 只要有就用 (除了对手手牌极少时)
        if state['my_skills'][4] > 0 and sum(state['op_hand']) >= 3: return [54 + 4]


        # ==========================
        # 2. 复杂反应 (基于威胁等级)
        # ==========================
        
        op_count = sum(state['op_hand'])
        my_count = sum(state['my_hand'])
        
        if op_count - my_count <= -2:
            return self.pressing_advantage(state, r3, r5_real, r5_empty)
        elif op_count - my_count > -2:
            return self.defence(state, r3, r5_real, r5_empty)
    # ----------------------------------------------------
    # 分支逻辑
    # ----------------------------------------------------
    def pressing_advantage(self,s, r3, r5_real, r5_empty):
        # 蓝区无损攻击
        sq_act = self._get_square_attack(s, r3, r5_empty)
        if sq_act: return sq_act
        # 如果做不到，换牌或偷牌
        combo = self._try_combo(['cube'], 'ST', r3, r5_empty)
        if combo: return combo
        combo = self._try_combo(['0'], 'RUIN', r3, r5_empty)
        if combo: return combo

        return [60] # End the Turn
    
    def defence(self,s, r3, r5_real, r5_empty):
        if s['my_shield'] <= 1:
            if self._can_make(['24'], r3, r5_empty):
                act = self._get_make_action(['24'], r3, r5_empty)
                return [act, 54+3, 54+3]
            
            sq_act = self._get_square_attack(s, r3, r5_empty)
            if sq_act: return sq_act

            return [61] # End the Round
        else:
            combo = self._try_combo(['cube'], 'ST', r3, r5_empty)
            if combo: return combo
            sq_act = self._get_square_attack(s, r3, r5_empty)
            if sq_act: return sq_act
            return [61] # End the Round
        
    # ----------------------------------------------------
    # 辅助工具
    # ----------------------------------------------------

    def _find_lethal(self, state, r3, r5_real):
        hp = state['op_hp']
        if state['op_shield'] > 0: return None
        if hp <= 50:
            act = self._get_attack_red(state, r3, r5_real)
            if act: return act
        elif hp <= 30:
            act = self._get_attack_yellow(state, r3, r5_real)
            if act: return act
        elif hp <= 10:
            act = self._get_attack_blue(state, r3, r5_real)
            if act: return act
        return None

    def _get_square_attack(self, state, r3, r5_set):
        """优先蓝/黄区平方，其次普通平方"""
        # 1. 蓝/黄
        act = self._get_zone_square_action(state, r3, r5_set)
        if act: return [act, 54+2]
        # 2. 普通
        combo = self._try_combo(['square'], 'DR', r3, r5_set)
        return combo

    def _get_zone_square_action(self, state, r3, r5_set):
        squares = [4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169]
        valid = [x for x in squares if x in (state['zones']['blue'] + state['zones']['yellow'])]
        for t in valid:
            act = self._get_action_for_target(t, r3, r5_set, is_zone=True, state=state)
            if act is not None: return act
        return None

    def _get_attack_red(self, state, r3, r5_real):
        for t in state['zones']['red']:
            act = self._get_action_for_target(t, r3, r5_real, is_zone=True, state=state)
            if act is not None: return act
        return None
    
    def _get_attack_yellow(self, state, r3, r5_real):
        for t in state['zones']['yellow'] + state['zones']['blue']:
            act = self._get_action_for_target(t, r3, r5_real, is_zone=True, state=state)
            if act is not None: return act
        return None
    
    def _get_attack_blue(self, state, r3, r5_real):
        for t in state['zones']['blue']:
            act = self._get_action_for_target(t, r3, r5_real, is_zone=True, state=state)
            if act is not None: return act
        return None
    
    def _try_combo(self, types, skill_name, r3, r5_set):
        act = self._get_make_action(types, r3, r5_set)
        if act is not None:
            skill_idx = self.skill_type_map[skill_name]
            return [act, 54 + skill_idx]
        return None

    def _can_make(self, types, r3, r5_set):
        return self._get_make_action(types, r3, r5_set) is not None

    def _get_make_action(self, types, r3, r5_set):
        target_list = []
        if 'cube' in types: target_list.extend([8, 27, 64])
        if 'square' in types: target_list.extend([4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169])
        if 'factorial' in types: target_list.extend([2, 6, 120])
        if '24' in types: target_list.append(24)
        if '1' in types: target_list.append(1)
        if '0' in types: target_list.append(0)
        
        for t in target_list:
            act = self._get_action_for_target(t, r3, r5_set)
            if act is not None: return act
        return None

    def _get_action_for_target(self, target, r3, r5_set, is_zone=False, state=None):
        eco_act = None
        std_act = None
        if is_zone:
            all_zones = state['zones']['red'] + state['zones']['yellow'] + state['zones']['blue']
            try:
                idx = all_zones.index(target)
                eco_act = 2 * idx
                std_act = 2 * idx + 1
            except ValueError: return None
        else:
            if target not in self.special_num_map: return None
            idx = self.special_num_map[target]
            eco_act = 14 + 2 * idx
            std_act = 14 + 2 * idx + 1

        if target in r3: return eco_act
        if target in r5_set: return std_act
        return None