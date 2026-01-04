# rename_remeta.py
from pathlib import Path
import pickle

def rename_and_remeta():
    # 自动获取脚本所在目录
    pool_dir = Path(__file__).parent
    
    # 安全验证
    if pool_dir.name != 'opponent_pool':
        print(f"错误：必须在 opponent_pool 文件夹中运行此脚本！")
        print(f"当前文件夹：{pool_dir.name}")
        print(f"完整路径：{pool_dir.absolute()}")
        return
    
    # 查找模型文件
    pth_files = sorted(pool_dir.glob("opponent_*.pth"))
    
    if not pth_files:
        print(f"未发现任何 .pth 文件")
        print(f"搜索路径：{pool_dir.absolute()}")
        return
    
    print(f"✓ 发现 {len(pth_files)} 个模型文件")
    
    # 新编号从 800000 开始，避免与新训练冲突
    new_start = 4000000
    
    for old_path in pth_files:
        old_name = old_path.name
        
        # 提取旧编号
        try:
            old_num = int(old_name.split('_')[1].split('.')[0])
        except:
            print(f"  跳过无效文件名: {old_name}")
            continue
        
        # 生成新文件名
        new_num = new_start + old_num     # 防止无上限
        new_path = pool_dir / f"opponent_{new_num:06d}.pth"
        meta_path = old_path.with_suffix('.meta')
        new_meta_path = new_path.with_suffix('.meta')
        
        # 重命名模型文件
        if old_path != new_path:
            old_path.rename(new_path)
            print(f"  重命名: {old_name} → {new_path.name}")
        
        # 创建或更新元数据
        meta = {'win_rate': 0.5, 'uses': 0, 'age': 0}
        
        # 如果旧meta存在，保留胜率
        if meta_path.exists():
            try:
                with open(meta_path, 'rb') as f:
                    old_meta = pickle.load(f)
                meta['win_rate'] = old_meta.get('win_rate', 0.5)
                print(f"    保留旧胜率: {meta['win_rate']:.2%}")
            except Exception as e:
                print(f"    读取旧meta失败: {e}")
        
        # 保存新meta
        with open(new_meta_path, 'wb') as f:
            pickle.dump(meta, f)
        
        # 删除旧meta文件
        if meta_path.exists() and meta_path != new_meta_path:
            meta_path.unlink()
    
    print(f"\n全部完成！新编号范围：{new_start}-{new_start+old_num}")
    print(f"所有模型已配备.meta文件，可以正常加载")

if __name__ == "__main__":
    rename_and_remeta()