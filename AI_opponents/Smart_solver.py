import numpy as np
import itertools
import time
from typing import List, Dict, Optional, Tuple, Set

class FastTemplateSolver:
    """
    Solver: 求解器
    - Forward Pass 缓存机制 (get_reachable_sets)，大幅加速 Mask 计算
    - 基于 NumPy 的矢量化计算，提升运算速度，努力优化保证训练速度
    解释一下为什么用float64：之前用float32一直和环境有数值不对齐的幽灵bug，不得已改为64
    """
    def __init__(self):
        self.np_add = np.add
        self.np_sub = np.subtract
        self.np_mul = np.multiply
        self.stats = {'calls': 0, 'hits': 0, 'total_time': 0.0}

    def solve(self, hand_counts: Dict[int, int], target: int, 
              min_cards: int = 3, max_cards: int = 5, 
              preferred_symbol: Optional[int] = None) -> Optional[List[int]]:
        """
        
        统一入口
        :param hand_counts: 手牌统计 {card_val: count}
        :param target: 目标数字
        :param min_cards: 最小卡牌数 (通常 3)
        :param max_cards: 最大卡牌数 (通常 5)
        :param preferred_symbol: 优先使用的符号 ID (如 14 代表 +). Solver 会尝试优先返回包含此符号的解.
        :return: 动作序列 (Card Indices) + [1000] (Confirm)
        """
        t0 = time.time()
        self.stats['calls'] += 1

        nums = []
        ops = []
        has_brackets = (hand_counts.get(18, 0) > 0 and hand_counts.get(19, 0) > 0)
        
        for k, v in hand_counts.items():
            if k <= 13: nums.extend([k] * v)
            elif 14 <= k <= 17: ops.extend([k] * v)

        # 优先符号逻辑
        if preferred_symbol is None:
            if 17 in ops and hand_counts[17] > 0:   # 优先使用/，因为能做除法运算少
                preferred_symbol = 17
            else:
                preferred_symbol = max((key for key in hand_counts if 14 <= key <= 17), 
                           key=hand_counts.get, default=None)
        
        # 排序保证确定性
        nums = sorted(nums)
        ops = sorted(ops)
        result = None

        # 3张牌搜索
        if min_cards <= 3 <= max_cards:
            # (N)=N 规则
            if has_brackets and target <= 13 and hand_counts.get(target, 0) > 0:
                result = [18, target, 19]
            
            if result is None:
                result = self._solve_len_3_numpy(nums, ops, target, preferred_symbol)
        
        # 5张牌搜索
        if result is None and min_cards <= 5 <= max_cards:
            result = self._solve_len_5_numpy(nums, ops, target, preferred_symbol)

        self.stats['total_time'] += (time.time() - t0)
        
        if result:
            self.stats['hits'] += 1
            return result + [1000] # Append Confirm
        return None

    def get_reachable_sets(self, hand_counts: Dict[int, int]) -> Tuple[Set[int], Set[int]]:
        """
        一次性计算所有可达数字
        返回: (reachable_3_cards, reachable_5_cards)
        """
        nums = []
        ops = []
        has_brackets = (hand_counts.get(18, 0) > 0 and hand_counts.get(19, 0) > 0)
        
        for k, v in hand_counts.items():
            if k <= 13: nums.extend([k] * v)
            elif 14 <= k <= 17: ops.extend([k] * v)
            
        reachable_3 = set()
        reachable_5 = set()
        
        # 1. 计算 3张牌可达集
        if len(nums) >= 2 and len(ops) >= 1:
            # (N) = N 特殊规则
            if has_brackets:
                reachable_3.update(set(nums))
                
            # 常规 A op B
            unique_pairs = sorted(list(set(itertools.permutations(nums, 2))))
            pairs_arr = np.array(unique_pairs, dtype=np.float32)
            if len(pairs_arr) > 0:
                col_a, col_b = pairs_arr[:, 0], pairs_arr[:, 1]
                unique_ops = sorted(list(set(ops)))
                
                for op in unique_ops:
                    res = self._calc_vec(col_a, col_b, op)
                    self._filter_and_add(res, reachable_3)

        # 2. 计算 5张牌可达集
        if len(nums) >= 3 and len(ops) >= 2:
            unique_triplets = sorted(list(set(itertools.permutations(nums, 3))))
            triplets_arr = np.array(unique_triplets, dtype=np.float32)
            
            if len(triplets_arr) > 0:
                col_a, col_b, col_c = triplets_arr[:, 0], triplets_arr[:, 1], triplets_arr[:, 2]
                unique_ops = sorted(list(set(ops)))
                op_pairs = list(itertools.product(unique_ops, repeat=2))
                
                for op1, op2 in op_pairs:
                    # 资源检查
                    needed = {op1: 0, op2: 0}
                    needed[op1] += 1; needed[op2] += 1
                    if any(ops.count(k) < v for k, v in needed.items()): continue

                    # Case 1: (A op1 B) op2 C  (仅当 op1 >= op2 时合法)
                    if self._priority(op1) >= self._priority(op2):
                        res1 = self._calc_vec(col_a, col_b, op1)
                        final_res = self._calc_vec(res1, col_c, op2)
                        valid_mask = (final_res > -9000) & np.isfinite(final_res)
                        self._filter_and_add(final_res, reachable_5)

                    # Case 2: A op1 (B op2 C)  (仅当 op2 > op1 时合法)
                    if self._priority(op2) > self._priority(op1):
                        res2 = self._calc_vec(col_b, col_c, op2)
                        final_res_right = self._calc_vec(col_a, res2, op1)
                        self._filter_and_add(final_res_right, reachable_5)
                        
        return reachable_3, reachable_5
    
    def _filter_and_add(self, res_array: np.ndarray, target_set: Set[int]):
        """
        核心修正逻辑:
        只保留那些与最近整数的距离 < 1e-5 的结果。
        这保证了 Mask=1 时，Solve 里的 isclose(target, atol=1e-5) 一定能通过。
        """
        # 1. 过滤无效值 (NaN, Inf, -9999)
        valid_mask = (res_array > -9000) & np.isfinite(res_array)
        valid_res = res_array[valid_mask]
        
        if valid_res.size == 0: return

        # 2. 找最近的整数
        nearest_int = np.rint(valid_res) # Round to nearest int
        
        # 3. 计算误差
        diff = np.abs(valid_res - nearest_int)
        
        # 4. 严格筛选: 误差必须小于 Solver 的容差 (1e-5)
        strict_mask = (diff < 1e-5)
        
        # 5. 存入 Set (转为 int)
        # 这样存进去的数字，Solver 绝对能找到
        valid_integers = nearest_int[strict_mask].astype(int)
        target_set.update(valid_integers)

    def _solve_len_3_numpy(self, nums, ops, target, pref_op):
        if len(nums) < 2 or len(ops) < 1: return None
        # 使用 sorted 保证确定性
        unique_pairs = sorted(list(set(itertools.permutations(nums, 2))))
        if not unique_pairs: return None
        
        pairs_arr = np.array(unique_pairs, dtype=np.float32)
        col_a, col_b = pairs_arr[:, 0], pairs_arr[:, 1]
        unique_ops = sorted(list(set(ops)))
        
        # 优先搜索逻辑
        search_phases = []
        if pref_op and pref_op in unique_ops:
            search_phases.append([pref_op])
            remaining = [o for o in unique_ops if o != pref_op]
            if remaining: search_phases.append(remaining)
        else:
            search_phases.append(unique_ops)
            
        for phase_ops in search_phases:
            for op_code in phase_ops:
                res = self._calc_vec(col_a, col_b, op_code)
                matches = np.where(np.abs(res - target) < 1e-5)[0]
                if matches.size > 0:
                    idx = matches[0] # 3张牌无优先级问题，取第一个即可
                    val_a = int(unique_pairs[idx][0])
                    val_b = int(unique_pairs[idx][1])
                    return [val_a, op_code, val_b]
        return None

    def _solve_len_5_numpy(self, nums, ops, target, pref_op):
        if len(nums) < 3 or len(ops) < 2: return None
        
        # 使用 sorted 保证确定性
        unique_triplets = sorted(list(set(itertools.permutations(nums, 3))))
        if not unique_triplets: return None
        
        triplets_arr = np.array(unique_triplets, dtype=np.float32)
        col_a, col_b, col_c = triplets_arr[:, 0], triplets_arr[:, 1], triplets_arr[:, 2]

        unique_ops = sorted(list(set(ops)))
        op_pairs = list(itertools.product(unique_ops, repeat=2))
        
        if pref_op:
            op_pairs.sort(key=lambda x: 0 if (x[0] == pref_op or x[1] == pref_op) else 1)

        for op1, op2 in op_pairs:
            needed_ops = {op1: 0, op2: 0}
            needed_ops[op1] += 1; needed_ops[op2] += 1
            if any(ops.count(k) < v for k, v in needed_ops.items()): continue

            # Case 1: (A op1 B) op2 C
            res1 = self._calc_vec(col_a, col_b, op1)
            final_res = self._calc_vec(res1, col_c, op2)
            matches = np.where(np.abs(final_res - target) < 1e-5)[0]
            
            # 遍历所有匹配项，寻找合法的优先级结构
            if matches.size > 0:
                if self._priority(op1) >= self._priority(op2):
                    # 只要有一个匹配，因为结构合法，直接返回第一个即可
                    # 之前的 Bug 是因为只看了 idx=0 可能是不合法的结构
                    # 这里先判断结构合法性，再看 matches 是否有值，逻辑是安全的
                    # 但为了保险，还是取 matches[0]
                    idx = matches[0]
                    vals = unique_triplets[idx]
                    return [vals[0], op1, vals[1], op2, vals[2]]

            # Case 2: A op1 (B op2 C)
            res2 = self._calc_vec(col_b, col_c, op2)
            final_res_right = self._calc_vec(col_a, res2, op1)
            matches_right = np.where(np.abs(final_res_right - target) < 1e-5)[0]
            
            if matches_right.size > 0:
                if self._priority(op2) > self._priority(op1):
                    idx = matches_right[0]
                    vals = unique_triplets[idx]
                    return [vals[0], op1, vals[1], op2, vals[2]]
        return None

    def _calc_vec(self, a, b, op):
        if op == 14: return self.np_add(a, b)
        if op == 15: return self.np_sub(a, b)
        if op == 16: return self.np_mul(a, b)
        if op == 17:
            with np.errstate(divide='ignore', invalid='ignore'):
                res = np.true_divide(a, b)
                res[~np.isfinite(res)] = -9999.0
                return res
        return a

    def _priority(self, op):
        if op in [16, 17]: return 2
        return 1
    
# 使用示例
if __name__ == "__main__":
    solver = FastTemplateSolver()
    
    # 场景: 3张牌凑 6，手里有 [2, 3, 1] 和 [+, *] (14, 16)
    # 期望优先用 +(14)
    hand = {1:1, 2:1, 3:2, 14:2, 16:1} # 两张3，两张+，一张*
    
    print("Test 1: 3 cards target 6 (Prioritize +)")
    # 2 * 3 = 6 (不含+), 3 + 3 = 6 (含+)
    # 应该返回 3 + 3
    sol = solver.solve(hand, 6, min_cards=3, max_cards=3, preferred_symbol=14)
    print(f"Solution: {sol}") # 预期: [3, 14, 3, 1000]

    print("\nTest 2: 5 cards target 23 (3 + 4 * 5)")
    hand2 = {2:1,3:1, 4:1, 5:1, 10:1, 14:1, 16:1} # 3,4,5, +, *
    sol2 = solver.solve(hand2, 23, min_cards=5, max_cards=5)
    print(f"Solution: {sol2}") # 预期: [3, 14, 4, 16, 5, 1000]
    
    print("\nTest 3: Bracket (3+3)=6")
    hand3 = {3:2, 14:1, 18:1, 19:1} # 3, 3, +, (, )
    # 强制用 5 张牌 (包含括号)
    # 这里的例子通常是 3+3=6。
    # 如果要测新规则 (9)=9:
    hand_special = {9:1, 18:1, 19:1}
    sol_special = solver.solve(hand_special, 9, min_cards=3, max_cards=5)
    print(f"Test 4: Special Rule (9)=9 -> Solution: {sol_special}")