You could see the English version below the Chinese vesion.
# CALC WARS文件使用与查阅README

#### 我们非常高兴您可以看到这里！这份README文件是关于CALC WARS项目的代码说明文件，应该会位于与CALC WARS主文件夹同一目录下。我们必须说明的是，文件中含有一份Project Report，这份报告非常详细地记录了我们本项目所使用的主要技术细节与主要成果。我们非常推荐您阅读。

### 如果您想要完整使用这个项目成果，我们建议您直接克隆整个仓库。
### 以下是CALC WARS主文件夹各个文件用途：

#### CALC_WARS.py: 核心游戏文件。如果您想游玩本游戏，请直接启动该文件并且确保已经安装了所有必要的库(如Pygame Numpy等等)。
#### Game_Rules.md: 规则介绍文件。您可以阅读此文件了解游戏规则。
#### CALC_WARS_AI.py 与 CALC_WARS_classic.py: 游戏AI对战模式与经典模式实现文件。您可以详细查看我们的游戏逻辑实现。
#### Model_checkpoints: 模型历史检查点储存文件。我们为游戏不同AI难度接入了不同阶段检查点文件，如果您想尝试其他检查点模型性能，您也可以在CALC_WARS.py主文件中修改文件路径。实际上检查点的末两位数字是编号，总共35个，但是考虑到文件大小和实际需要，您最终应该会发现我们只留下8个检查点。
#### AI_opponents: AI模型训练文件。以下是更详细说明：

##### PPO_Command_Res.py: 核心训练文件。包含模型架构设计与训练流程pipeline。
##### Env_sparse.py: 模型模拟游戏训练环境。
##### Smart_solver.py: 训练所需的符号求解器文件，也是模型分层架构的底层部分。
##### Demonbot_x.py系列文件：训练所需决策树专家文件。如果您对这些决策树专家的策略感兴趣，可以查看其本体或对应的DecisionTree文件(如果存在)。
##### Arena.py: 模型测评性能文件。
##### 其余文件(WandB_upload.py等等): 用于debug或日志上传文件。

#### opponent_pool.py文件: 历史版本对手池管理文件。
#### sounds.py: 游戏声音文件。
#### wandb: 训练日志上传Weights & Biases文件。如果您对我们的训练日志感兴趣，可以用WandB运行这些文件查看(我们不能展示我们的WandB账号APIkey)。
#### BAckground.png 与 Shield.png: 游戏图标文件。


# CALC WARS: File Usage and Reference README

#### We are glad you are here!:) This README provides technical documentation for the CALC WARS project and should be located in the same directory as the main CALC WARS folder. We must demonstrate that there's a Project Report in this folder, which summarize the main techniques and results of the projevct. We highly recommand you read it.

### Purpose of Files in the CALC WARS Directory:

#### CALC_WARS.py: 
The core game file. To play the game, run this file directly. Ensure all necessary libraries (e.g., Pygame, Numpy, etc.) are installed.

#### Game_Rules.md: 
The rules documentation. Read this file to understand the game mechanics and rules.

#### CALC_WARS_AI.py & CALC_WARS_classic.py: 
Implementation files for the "AI Battle Mode" and "Classic Mode." You can examine these for detailed game logic implementation.

#### Model_checkpoints: 
Storage for historical model checkpoints. Different checkpoints are used for various AI difficulty levels. If you wish to test the performance of other checkpoints, you can modify the file paths in the `CALC_WARS.py` main file.

#### AI_opponents: 
AI model training files. Detailed breakdown:

##### PPO_Command_Res.py: 
Core training file. Contains the model architecture design and the training pipeline.

##### Env_sparse.py: 
The simulated game training environment for the model.

##### Smart_solver.py: 
The symbolic solver required for training, which serves as the low-level component of the model's hierarchical architecture.

##### Demonbot_x.py series: 
Decision Tree Expert files used for training. If you are interested in the strategies of these experts, you can check the files directly or their corresponding DecisionTree files (if available).

##### Arena.py: 
Performance evaluation file for the models.

##### Miscellaneous (WandB_upload.py, etc.): 
Files used for debugging or uploading logs.

#### opponent_pool.py: 
Management file for the historical version opponent pool.

#### sounds.py: 
Game sound effect files.

#### wandb: 
Weights & Biases (WandB) files for training logs. If you are interested in our training logs, you can run these files with WandB (note: our private API keys are not provided).

#### Background.png & Shield.png: 
Game asset/icon files.
