import numpy as np
from collections import deque
from Smart_solver import FastTemplateSolver

class DemonAgent_V2:
    """
    二代恶魔机器人 (Berserker / Ruin Archetype)
    策略核心：进攻与破坏 (Aggression & Disruption)
    特点：
    1. 极其厌恶 End Turn (Action 60)，倾向于 End Round (Action 61) 以争抢下一轮先手。
    2. 不进行任何防守 (不主动造盾)。
    3. 优先破坏对手资源 (RUIN/STEAL/PIERCE)。
    """
    def __init__(self, logger=None):
        self.solver = FastTemplateSolver()
        self.logger = logger
        self.name = 'Demon_v2'
        # [核心] 内部动作缓存队列
        self.action_queue = deque()
        
        # 特殊数字映射表
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
        # 1. 执行队列
        if self.action_queue:
            return self.action_queue.popleft()

        # 2. 解析状态
        state = self._parse_observation(observation)
        
        # 3. 决策
        plan = self._evaluate_decision_tree(state)
        
        if not plan:
            # 兜底逻辑：二代恶魔倾向于结束本轮 (61)，而不是结束回合 (60)
            # 只有在规则不允许结束本轮时（例如还没行动过），才被迫结束回合
            # 但根据决策树描述，它是“结束本轮”的狂热者
            # 这里做一个简单映射：优先 61，不行则 60
            plan = [61] 
            
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
        
        # 二代恶魔为了进攻和破坏，允许在必要时使用5张牌 (Aggressive)
        # 但为了保持和决策树的一致性（通常没特殊说明用3张），我们默认用 r5_empty
        # 除非是斩杀或红区
        r5_empty = set() 

        # ==========================
        # 1. 基础本能反应 (Basic Instinct)
        # ==========================
        
        # [斩杀]
        if state['op_shield'] == 0:
            lethal_act = self._find_lethal(state, r3, r5_real)
            if lethal_act: return [lethal_act]

        # [HEAL] 血量 <= 100
        if state['my_hp'] <= 100 and state['my_skills'][0] > 0: return [54 + 0] 
        
        # [SHIELD] 护盾 == 0
        if state['my_shield'] == 0 and state['my_skills'][3] > 0: return [54 + 3]
        
        # [RUIN] 对手牌 >= 3
        if state['my_skills'][4] > 0 and sum(state['op_hand']) >= 3: return [54 + 4]


        # ==========================
        # 2. 复杂反应 (Complex Reaction)
        # ==========================
        
        my_count = sum(state['my_hand'])
        op_count = sum(state['op_hand'])
        diff_count = my_count - op_count
        
        # 情况1: 我方资源 >= 对方 (Advantage)
        if diff_count >= 0:
            if diff_count <= 6:
                return self._case_1_small_advantage(state, r3, r5_real, r5_empty)
            else:
                return self._case_1_large_advantage(state, r3, r5_real, r5_empty)
        
        # 情况2: 我方资源 < 对方 (Disadvantage)
        else:
            return self._case_2_disadvantage(state, r3, r5_real, r5_empty)

    # ----------------------------------------------------
    # 分支实现
    # ----------------------------------------------------

    def _case_1_small_advantage(self, state, r3, r5_real, r5_empty):
        # 1. 凑立方 -> ST
        combo = self._try_combo(['cube'], 'ST', r3, r5_empty)
        if combo: return combo
        
        # 2. 无法凑立方
        if state['op_shield'] > 0:
            # 有盾：蓝/黄平方 -> DR (无损)
            act = self._get_zone_square_action(state, r3, r5_empty)
            if act: return [act, 54+2]
            
            # 1 -> PI
            combo = self._try_combo(['1'], 'PI', r3, r5_empty)
            if combo: return combo
            
            # 普通平方 -> DR
            combo = self._try_combo(['square'], 'DR', r3, r5_empty)
            if combo: return combo
            
            # 0 -> RU
            combo = self._try_combo(['0'], 'RUIN', r3, r5_empty)
            if combo: return combo
            
            return [61] # End Round

        else:
            # 无盾
            # 红/黄区 (允许 r5_real 进攻红区)
            act = self._get_attack_action(state, r3, r5_real)
            if act: return [act]
            
            # 平方 -> DR
            combo = self._try_combo(['square'], 'DR', r3, r5_empty)
            if combo: return combo
            
            return [61] # End Round

    def _case_1_large_advantage(self, state, r3, r5_real, r5_empty):
        if state['op_shield'] > 0:
            # 有盾
            act = self._get_zone_square_action(state, r3, r5_empty)
            if act: return [act, 54+2]
            
            combo = self._try_combo(['1'], 'PI', r3, r5_empty)
            if combo: return combo
            
            combo = self._try_combo(['square'], 'DR', r3, r5_empty)
            if combo: return combo
            
            combo = self._try_combo(['0'], 'RUIN', r3, r5_empty)
            if combo: return combo
            
            return [61]
        else:
            # 无盾
            act = self._get_attack_action(state, r3, r5_real)
            if act: return [act]
            
            combo = self._try_combo(['square'], 'DR', r3, r5_empty)
            if combo: return combo
            
            combo = self._try_combo(['0'], 'RUIN', r3, r5_empty)
            if combo: return combo
            
            return [61]

    def _case_2_disadvantage(self, state, r3, r5_real, r5_empty):
        # 劣势疯狂进攻
        combo = self._try_combo(['cube'], 'ST', r3, r5_empty)
        if combo: return combo
        
        if state['op_shield'] > 0:
            act = self._get_zone_square_action(state, r3, r5_empty)
            if act: return [act, 54+2]
            
            combo = self._try_combo(['1'], 'PI', r3, r5_empty)
            if combo: return combo
            
            combo = self._try_combo(['square'], 'DR', r3, r5_empty)
            if combo: return combo
            
            return [61]
        else:
            act = self._get_attack_action(state, r3, r5_real)
            if act: return [act]
            
            combo = self._try_combo(['square'], 'DR', r3, r5_empty)
            if combo: return combo
            
            return [61]

    # ----------------------------------------------------
    # 辅助工具
    # ----------------------------------------------------

    def _find_lethal(self, state, r3, r5_real):
        hp = state['op_hp']
        # 只有对手无盾才考虑斩杀 (基础本能里有判断，但双重保险)
        if state['op_shield'] > 0: return None
        
        # 优先红区
        if hp <= 50:
            for t in state['zones']['red']:
                act = self._get_action_for_target(t, r3, r5_real, is_zone=True, state=state)
                if act: return act
        # 其次黄区
        if hp <= 30:
            for t in state['zones']['yellow']:
                act = self._get_action_for_target(t, r3, r5_real, is_zone=True, state=state)
                if act: return act
        # 最后蓝区
        if hp <= 10:
             for t in state['zones']['blue']:
                act = self._get_action_for_target(t, r3, r5_real, is_zone=True, state=state)
                if act: return act
        return None

    def _get_zone_square_action(self, state, r3, r5_set):
        # 寻找属于蓝区或黄区的平方数
        squares = [4, 9, 16, 25, 36, 49, 81, 100, 121, 144, 169]
        valid_targets = [x for x in squares if x in (state['zones']['blue'] + state['zones']['yellow'])]
        for t in valid_targets:
            act = self._get_action_for_target(t, r3, r5_set, is_zone=True, state=state)
            if act is not None: return act
        return None

    def _get_attack_action(self, state, r3, r5_real):
        # 优先红区 (允许5张)
        for t in state['zones']['red']:
            act = self._get_action_for_target(t, r3, r5_real, is_zone=True, state=state)
            if act is not None: return act
        # 其次黄区 (限制3张, 除非有特殊逻辑，这里沿用 r3/empty)
        # 二代决策树没明确说黄区能不能用5张，但为了进攻性，我们假设红区最重要
        for t in state['zones']['yellow']:
            act = self._get_action_for_target(t, r3, set(), is_zone=True, state=state)
            if act is not None: return act
        return None

    def _try_combo(self, types, skill_name, r3, r5_set):
        act = self._get_make_action(types, r3, r5_set)
        if act is not None:
            skill_idx = self.skill_type_map[skill_name]
            return [act, 54 + skill_idx]
        return None

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
            except ValueError:
                return None
        else:
            if target not in self.special_num_map: return None
            idx = self.special_num_map[target]
            eco_act = 14 + 2 * idx
            std_act = 14 + 2 * idx + 1

        if target in r3: return eco_act
        if target in r5_set: return std_act
        return None