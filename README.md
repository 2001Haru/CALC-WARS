Chinese version first. English version is provided below.

# CALC WARS 文件使用与查阅 README

## 欢迎与说明

> 🌟 核心特色：强化学习训练 AI 对手（Reinforcement Learning-based AI Opponents）
>
> ⚔️ 该项目不仅是一个可玩的数学策略游戏，同时也是一个可训练、可评测、可迭代的 AI 对战系统。

欢迎来到 CALC WARS 项目。此 README 是项目主目录中的代码与资源说明文档，通常位于 CALC WARS 主文件夹同级目录下。

同时，本目录内包含一份 Project Report。该报告更系统地记录了本项目采用的关键技术细节、设计方案与主要成果，强烈建议您配合本 README 一并阅读。

如果您希望完整使用本项目成果，建议直接克隆整个仓库，以避免遗漏模型、资源文件或历史训练组件。

## 快速导航

- 想直接开始游戏：请先看 `CALC_WARS.py`
- 想理解规则与玩法：请阅读 `Game_Rules.md`
- 想查看模式实现与逻辑：请查看 `CALC_WARS_AI.py` 与 `CALC_WARS_classic.py`
- 想研究训练流程与模型：请查看 `AI_opponents/` 与 `Model_checkpoints/`

## 🚀 快速开始（命令块）

### 1) 克隆项目

```bash
git clone https://github.com/2001Haru/CALC-WARS
cd CALC-WARS
```

### 2) 安装依赖（游戏运行）

```bash
pip install pygame numpy
```

### 3) 启动游戏

```bash
python "CALC WARS Game/CALC_WARS_main/CALC_WARS.py"
```

现在你可以享受这个卡牌对战游戏，体验我们训练的AI对手！

## CALC WARS 主目录文件与用途

### 核心游戏与规则

#### `CALC_WARS.py`
核心游戏启动文件。如果您希望直接游玩，请运行该文件，并确保已安装必要依赖（例如 Pygame、Numpy 等）。

#### `Game_Rules.md`
规则介绍文件。用于了解完整游戏机制、操作方式与胜负条件。

#### `CALC_WARS_AI.py` 与 `CALC_WARS_classic.py`
分别对应 AI 对战模式与经典模式的实现文件。若您希望深入了解游戏逻辑，这两个文件是最直接的入口。

### 模型与训练相关

#### `Model_checkpoints/`
模型历史检查点存储目录。项目为不同 AI 难度接入了不同阶段的检查点；如果您想测试其他检查点性能，可在 `CALC_WARS.py` 中调整加载路径。

补充说明：检查点文件名末两位数字用于编号。原始总量为 35 个；考虑文件体积与实际使用需求，最终保留了 8 个代表性检查点。

#### `AI_opponents/`
AI 模型训练相关目录，包含训练环境、求解器、专家策略与评测脚本。主要文件如下：

##### `PPO_Command_Res.py`
核心训练文件，包含模型架构设计与训练流程 pipeline。

##### `Env_sparse.py`
模型训练使用的模拟游戏环境。

##### `Smart_solver.py`
训练所需符号求解器文件，也是分层架构中的底层关键组件。

##### `Demonbot_x.py` 系列
训练所需的决策树专家文件。如果您希望研究专家策略，可查看对应本体文件或其 DecisionTree 文件（若存在）。

##### `Arena.py`
模型性能评测文件，用于测试和比较模型表现。

##### 其余文件（如 `WandB_upload.py` 等）
主要用于调试、日志处理或训练记录上传。

### 其他项目资源

#### `opponent_pool.py`
历史版本对手池管理文件。

#### `sounds.py`
游戏音效相关文件。

#### `wandb/`
训练日志上传到 Weights & Biases（WandB）的相关文件。如果您对训练日志感兴趣，可使用 WandB 运行与查看；出于安全原因，仓库不包含私有 API Key。

#### `Background.png` 与 `Shield.png`
游戏图标与素材资源文件。

---

# CALC WARS: File Usage and Reference README

## Welcome and Overview

> 🌟 Signature Feature: Reinforcement Learning-based AI Opponents
>
> ⚔️ This project is not only a playable math strategy game, but also a trainable, evaluable, and iterative AI battle system.

Welcome to the CALC WARS project. This README serves as a technical and structural reference for source files and assets, and is typically located alongside the main CALC WARS folder.

This directory also includes a Project Report. That report documents the key technical details, design decisions, and major outcomes of this project in depth, and is highly recommended reading together with this README.

If you want to use the full project deliverable, cloning the entire repository is strongly recommended to avoid missing model checkpoints, assets, or historical training components.

## Quick Navigation

- Want to launch the game quickly: start with `CALC_WARS.py`
- Want to understand rules and mechanics: read `Game_Rules.md`
- Want to inspect implementation logic: check `CALC_WARS_AI.py` and `CALC_WARS_classic.py`
- Want to study training and model workflow: review `AI_opponents/` and `Model_checkpoints/`

## 🚀 Quick Start (Command Blocks)

### 1) Clone the repository

```bash
git clone <your-repo-url>
cd <repo-root>
```

### 2) Install dependencies (for gameplay)

```bash
pip install pygame numpy
```

### 3) Launch the game

```bash
python "CALC WARS Game/CALC_WARS_main/CALC_WARS.py"
```

And that's our Game! Try our AI opponent!

## Purpose of Files in the CALC WARS Directory

### Core Gameplay and Rules

#### `CALC_WARS.py`
The core game entry file. To play the game directly, run this file and make sure required dependencies are installed (e.g., Pygame, Numpy, etc.).

#### `Game_Rules.md`
Rules documentation for understanding gameplay mechanics, controls, and win/loss conditions.

#### `CALC_WARS_AI.py` and `CALC_WARS_classic.py`
Implementation files for AI Battle Mode and Classic Mode. These are the primary files to inspect for detailed gameplay logic.

### Model and Training Components

#### `Model_checkpoints/`
Storage for historical model checkpoints. Different checkpoints are connected to different AI difficulty levels. If you want to test alternative checkpoints, you can modify the loading path in `CALC_WARS.py`.

Additional note: the last two digits of checkpoint names serve as IDs. The original set had 35 checkpoints in total; considering file size and practical usage, 8 representative checkpoints are retained in this release.

#### `AI_opponents/`
Directory for model training-related files, including environment simulation, symbolic solving, expert strategies, and evaluation scripts. Main files include:

##### `PPO_Command_Res.py`
Core training file containing model architecture design and the training pipeline.

##### `Env_sparse.py`
Simulated game environment used for model training.

##### `Smart_solver.py`
Symbolic solver required by training, serving as a foundational low-level component in the hierarchical architecture.

##### `Demonbot_x.py` series
Decision-tree expert files used during training. If you are interested in expert strategies, inspect these files directly or their corresponding DecisionTree files (when available).

##### `Arena.py`
Model performance evaluation file used to benchmark model behavior.

##### Miscellaneous files (e.g., `WandB_upload.py`)
Used for debugging, logging utilities, or training log uploads.

### Other Project Resources

#### `opponent_pool.py`
Management file for historical-version opponent pools.

#### `sounds.py`
Game sound effect related file.

#### `wandb/`
Weights & Biases (WandB) integration files for training logs. If you want to inspect training records, you can run these files with WandB. For security reasons, private API keys are not included.

#### `Background.png` and `Shield.png`
Game icon and visual asset files.
