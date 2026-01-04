import numpy as np
from Env_sparse import Game, CardType
from Demonbot import DemonAgent
import time

def debug_demon_logic():
    '''调试Demonbot决策逻辑，逐步打印其输入输出和关键中间变量'''
    # 初始化环境和魔王
    env = Game()
    demon = DemonAgent()
    
    # 强制让魔王做 Player 1 (方便观察)
    env.current_player = env.player1
    # 稍微作弊一下，给魔王塞几张好牌，测试它会不会用
    # 例如：给它塞个 3, 8 (可以凑24), 或者 4 (平方)
    # 这里先不改手牌，看自然发牌的情况
    
    print(f"初始状态: P1(魔王) HP={env.player1.hp}, P2(沙包) HP={env.player2.hp}")
    
    # 模拟运行 50 步
    for step in range(50):
        print(f"\n--- Step {step} ---")
        
        # 1. 检查此时轮到谁
        current_pid = 0 if env.current_player == env.player1 else 1
        print(f"当前行动: Player {current_pid + 1} ({env.current_player.name})")
        
        # 2. 如果是魔王 (P1)，深入检查它的输入数据
        if env.current_player == env.player1:
            hand = [i for i, c in enumerate(env.current_player.hand) if c > 0]
            hand_full_str = []
            for i in range(20):
                if env.current_player.hand[i] > 0:
                    count = env.current_player.hand[i]
                    name = str(i) if i < 14 else ['+','-','*','/','(',')'][i-14]
                    hand_full_str.append(f"{name}x{count}")
            
            print(f"[魔王视界] 手牌: {', '.join(hand_full_str)}")
            print(f"[魔王视界] 手牌总数: {sum(env.current_player.hand[:20])}")
            print(f"[魔王视界] 技能卡: {env.current_player.skill_cards[:6]}")
            
            # --- 关键检查点：Solver 是否工作？ ---
            hand_counts = {i: c for i, c in enumerate(env.current_player.hand) if c > 0}
            r3, r5 = env.solver.get_reachable_sets(hand_counts)
            print(f"[Solver检查] 3张可达集合大小: {len(r3)}")
            print(f"[Solver检查] 5张可达集合大小: {len(r5)}")
            
            if len(r3) > 0:
                print(f"   示例可达数字: {list(r3)[:5]}...")
            else:
                print("   [警告] Solver 返回空集！魔王以为自己没牌可出！")

            # --- 关键检查点：目标判定 ---
            zones = [env.target.red_zone] + env.target.yellow_zone + env.target.blue_zone
            print(f"[战场目标] 红:{env.target.red_zone}, 黄:{env.target.yellow_zone}, 蓝:{env.target.blue_zone}")
            
            hits = [z for z in zones if z in r3]
            print(f"[魔王计算] 能打到的区域: {hits}")
            
            # 3. 让魔王做决定
            try:
                action = demon.get_action(env.state_vector)
                action_name = "未知"
                if action == 60: action_name = "End Turn"
                elif action == 61: action_name = "End Round"
                elif 54 <= action <= 59: action_name = f"Skill {action-54}"
                elif action <= 53: 
                    # 反推是什么动作
                    is_std = (action % 2 != 0)
                    target_idx = action // 2
                    if target_idx < 7: action_name = f"Attack Zone {target_idx} ({'Std' if is_std else 'Eco'})"
                    else: action_name = f"Make Special {target_idx-7} ({'Std' if is_std else 'Eco'})"
                
                print(f"==> 魔王决定: Action {action} [{action_name}]")
                
                # 4. 检查是否合法 (Mask)
                mask = env.get_oracle_mask()
                if mask[action] == 0:
                    print(f"!!!!!! 致命错误: 魔王选择了一个被 Mask 禁止的动作！Mask[{action}] == 0")
                else:
                    print(f"    (动作合法)")
                    
            except Exception as e:
                print(f"!!!!!! 魔王大脑崩溃: {e}")
                import traceback
                traceback.print_exc()
                break

        else:
            # 简单的沙包对手，只 End Turn
            action = 60
            if env.player2_round_end or env.player1_round_end:
                action = 61
            print(f"==> 沙包决定: Action {action}")

        # 5. 执行环境步
        state, reward, done, info = env.step(action)
        print(f"    [结果] Reward: {reward}, Info: {info['message']}")
        
        if done:
            print("\n=== 游戏结束 ===")
            print(f"P1 HP: {env.player1.hp}, P2 HP: {env.player2.hp}")
            if env.player1.hp > 0: print("魔王赢了 (但这是否是意外？)")
            else: print("魔王输了")
            break
            
if __name__ == "__main__":
    debug_demon_logic()