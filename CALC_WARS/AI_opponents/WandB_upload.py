import wandb
import os
os.environ["WANDB_API_KEY"] = "aa28905ac52e47e17f29015051d62a5c4b71d186"

# 1. 初始化 API
api = wandb.Api()

# 2. 获取指定的 run 对象
# 将下面的路径替换为你实际的路径，例如 "your_username/Exp_damage_Teacher/3x9s8d7f"
run = api.run("linxi080530-westlake-university/Card-RL-Experiment/8ga4sfmi")

# 3. 上传文件
# 将 'env.py' 替换为你实际的环境文件路径
# 1. 定义你的文件路径 (保持你原来的写法)
file_path = r"D:\HALcode\Gameplay\CALC_WARS\AI_opponents\Env_exp_damage.py"

# 2. 获取该文件所在的文件夹路径
file_dir = os.path.dirname(file_path)

# 3. 上传文件，并指定 root 参数
# root=file_dir 的意思是：以文件所在的文件夹为基准。
# 这样 WandB 就会把文件当作 "Env_exp_damage.py" 上传，而不是 "AI_opponents\Env_exp_damage.py"，从而避开了反斜杠的问题。
run.upload_file(file_path, root=file_dir)

# 如果环境文件在一个文件夹里，比如 envs/custom_env.py
# run.upload_file("envs/custom_env.py")

print("文件上传成功！请刷新 Wandb 网页查看 Files 标签页。")