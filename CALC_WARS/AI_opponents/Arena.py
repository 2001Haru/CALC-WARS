"""
Arena for model-vs-script evaluation (inference only)

Usage example:
    python Arena.py --model ppo_311026_1.pth --games 3000 --device cpu --deterministic

This script:
- Loads a saved checkpoint (expects key 'policy_net_state_dict' or raw state_dict)
- Builds a CommanderNet and performs pure inference (no training)
- Plays N games vs each of DemonAgent, DemonAgent_V2, DemonAgent_V3
- Each game has max length (default 2048); if exceeded, winner decided by higher HP
- Prints wins / losses / draws and win rates

Requirements: run from repository root where modules in this folder are importable.
"""
import argparse
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from Env_sparse import Game
from PPO_Command_Res import CommanderNet
from Demonbot import DemonAgent
from Demonbot_2 import DemonAgent_V2
from Demonbot_3 import DemonAgent_V3
from Demonbot_4 import DemonAgent_V4
from Demonbot_5 import DemonAgent_V5
import sys

# ===== USER CONFIG: edit these values directly (no CLI needed) =====
# Fill your model path and evaluation parameters here
MODEL_PATH = r"D:\HALcode\Gameplay\CALC_WARS\ppo_369500_1.pth"  # <-- put your model file path here
GAMES_PER_OPPONENT = 3000                # number of games vs each demon
MAX_STEPS = 2048                         # max steps per game
DEVICE = "cpu"                          # 'cpu' or 'cuda'
DETERMINISTIC = True                     # True -> argmax, False -> sample
HIDDEN_SIZE = 512                        # network hidden size (match training)
NUM_BLOCKS = 8                           # network depth (match training)
SEED = None                              # optional random seed
# ================================================================


class ModelPlayer:
    def __init__(self, checkpoint_path: str, device: torch.device = torch.device('cpu'),
                 hidden_size: int = 512, num_blocks: int = 8, deterministic: bool = False):
        self.device = device
        self.deterministic = deterministic

        # build network
        self.net = CommanderNet(obs_dim=92, action_dim=62, hidden_size=hidden_size, num_blocks=num_blocks)
        self.net.to(self.device)

        # load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device,weights_only=False)
        state_dict = None
        if isinstance(ckpt, dict) and 'policy_net_state_dict' in ckpt:
            state_dict = ckpt['policy_net_state_dict']
        elif isinstance(ckpt, dict) and 'policy_net' in ckpt:
            state_dict = ckpt['policy_net']
        else:
            # assume ckpt is a raw state_dict
            state_dict = ckpt

        # attempt to load
        try:
            self.net.load_state_dict(state_dict)
        except Exception as e:
            # try tolerant loading
            print(f"Warning: strict load failed ({e}), trying non-strict load...")
            self.net.load_state_dict(state_dict, strict=False)

        self.net.eval()

    def get_action(self, obs: np.ndarray, mask: np.ndarray):
        """Return an int action given obs (np.array shape (92,)) and mask (62)
        mask contains 1.0 for legal actions, 0 otherwise.
        If deterministic, choose argmax over legal logits; otherwise sample.
        """
        # prepare tensors
        x = torch.from_numpy(obs.astype(np.float32)).to(self.device)
        mask_t = torch.from_numpy(mask.astype(np.float32)).to(self.device)

        with torch.no_grad():
            logits, _ = self.net(x)
            logits = logits.view(-1)  # shape (62,)

            # apply mask: set illegal logits to large negative
            legal = (mask_t > 0.5)
            if legal.sum().item() == 0:
                # no legal action (shouldn't happen often) -> End Round
                return 61

            neg_inf = -1e9
            safe_logits = torch.where(legal, logits, torch.tensor(neg_inf, device=self.device))

            if self.deterministic:
                act = int(safe_logits.argmax().item())
                return act
            else:
                probs = F.softmax(safe_logits, dim=0)
                # numerical safety: project onto legal actions only
                probs = probs * legal.float()
                s = probs.sum().item()
                if s <= 0:
                    # fallback to argmax
                    return int(safe_logits.argmax().item())
                probs = probs / s
                m = torch.distributions.Categorical(probs=probs)
                return int(m.sample().item())


def play_one_game(env: Game, student: ModelPlayer, demon_agent, max_steps: int = 2048, deterministic: bool = False):
    """Plays a single game until done or max_steps. Returns 1 if student wins, -1 demon wins, 0 draw."""
    env.reset()
    step = 0

    # initial obs
    obs = env._get_state()

    while step < max_steps:
        step += 1
        # who acts
        if env.current_player == env.player1:
            mask = env.get_oracle_mask()
            action = student.get_action(obs, mask)
        else:
            # demon expects observation vector same as env._get_state()
            action = demon_agent.get_action(obs)

        obs, rew, done, info = env.step(action)

        if done:
            # decide winner by HP
            p1_hp = env.player1.hp
            p2_hp = env.player2.hp
            if p1_hp > p2_hp:
                return 1
            elif p1_hp < p2_hp:
                return -1
            else:
                return 0

    # reached max_steps, use HP at last obs
    p1_hp = env.player1.hp
    p2_hp = env.player2.hp
    if p1_hp > p2_hp:
        return 1
    elif p1_hp < p2_hp:
        return -1
    else:
        return 0


def run_evaluation(model_path: str, games_per_opponent: int = 3000, max_steps: int = 2048,
                   device: str = 'cpu', deterministic: bool = False,
                   hidden_size: int = 512, num_blocks: int = 4, seed: int = None):

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    device_obj = torch.device(device if torch.cuda.is_available() and device.startswith('cuda') else 'cpu')
    student = ModelPlayer(model_path, device=device_obj, hidden_size=hidden_size, num_blocks=num_blocks,
                          deterministic=deterministic)

    opponents = [
        (DemonAgent_V5, "Demon_v5"),
        (DemonAgent, "Demon_v1"),
        (DemonAgent_V3, "Demon_v3"),
        (DemonAgent_V2, "Demon_v2"),
        (DemonAgent_V4, "Demon_v4"),
    ]

    results = {}

    for opp_cls, name in opponents:
        wins = 0
        losses = 0
        draws = 0
        start = time.time()
        print(f"\n=== Playing {games_per_opponent} games vs {name} (device={device_obj}) ===")
        for i in range(1, games_per_opponent + 1):
            env = Game()
            demon = opp_cls()
            res = play_one_game(env, student, demon, max_steps=max_steps, deterministic=deterministic)
            if res == 1:
                wins += 1
            elif res == -1:
                losses += 1
            else:
                draws += 1

            if i % 100 == 0 or i == games_per_opponent:
                elapsed = time.time() - start
                print(f"  {i}/{games_per_opponent}  wins:{wins} losses:{losses} draws:{draws} elapsed:{elapsed:.1f}s", end='\r')
        # final print newline
        print()
        total = games_per_opponent
        win_rate = wins / total
        loss_rate = losses / total
        draw_rate = draws / total
        results[name] = {
            'wins': wins, 'losses': losses, 'draws': draws,
            'win_rate': win_rate, 'loss_rate': loss_rate, 'draw_rate': draw_rate
        }
        print(f"Results vs {name}: wins {wins} / {total} (win rate {win_rate:.2%}), losses {losses}, draws {draws}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Arena evaluation (if no CLI args are provided, the embedded CONFIG values will be used).')
    parser.add_argument('--model', type=str, default=None, help='Path to checkpoint (.pth). If omitted, MODEL_PATH in this file is used')
    parser.add_argument('--games', type=int, default=None, help='Games per opponent (overrides file CONFIG)')
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--hidden_size', type=int, default=None)
    parser.add_argument('--num_blocks', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    # If the user didn't pass a model via CLI, prefer the file CONFIG
    if args.model is None:
        print("Using in-file CONFIG values (edit the CONFIG block at top of Arena.py to change settings)")
        model = MODEL_PATH
        games = GAMES_PER_OPPONENT
        max_steps = MAX_STEPS
        device = DEVICE
        deterministic = DETERMINISTIC or args.deterministic
        hidden_size = HIDDEN_SIZE
        num_blocks = NUM_BLOCKS
        seed = SEED
    else:
        model = args.model
        games = args.games if args.games is not None else GAMES_PER_OPPONENT
        max_steps = args.max_steps if args.max_steps is not None else MAX_STEPS
        device = args.device if args.device is not None else DEVICE
        deterministic = args.deterministic or DETERMINISTIC
        hidden_size = args.hidden_size if args.hidden_size is not None else HIDDEN_SIZE
        num_blocks = args.num_blocks if args.num_blocks is not None else NUM_BLOCKS
        seed = args.seed if args.seed is not None else SEED

    results = run_evaluation(
        model_path=model,
        games_per_opponent=games,
        max_steps=max_steps,
        device=device,
        deterministic=deterministic,
        hidden_size=hidden_size,
        num_blocks=num_blocks,
        seed=seed
    )

    print('\n==== Summary ===')
    for name, r in results.items():
        print(f"{name}: wins={r['wins']} losses={r['losses']} draws={r['draws']} win_rate={r['win_rate']:.2%}")


if __name__ == '__main__':
    main()
