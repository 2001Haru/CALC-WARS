#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
24点卡牌游戏启动脚本
"""

import sys
import os

def main():
    """启动游戏"""
    print("=" * 50)
    print("Welcom to 24 Points Game!")
    print("=" * 50)
    print()
    print("Game rules:")
    print("1. Each Player has 100 HP")
    print("2. Use your cards to calculate, at least use 3 cards one time")
    print("3. The result of calculation hit the different areas of the target causes different damages")
    print("4. Special result will be awarded Skill Cards")
    print("5. Damage the other player until no HP remaining!")
    print()
    print("Operation Instruction:")
    print("- Click the card to confirm/cancel")
    print("- Click 'Confirm' to calculate")
    print("- Click 'End' to end your turn")
    print("- Click Skill Cards to use them")
    print()
    
    try:
        from CALC_WARS import Game
        game = Game()
        print("Game starting...")
        game.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Please check the installation of pygame")

if __name__ == "__main__":
    main()

