"""使用 colmap-rerun 可视化 SfM 稀疏重建结果"""
from pathlib import Path
from colmap_rerun.core.reconstruction import load_sparse_model
from colmap_rerun.visualization.visualizer import visualize_reconstruction

# ==================== 可自定义：与 script.py 保持一致 ====================
DATASET = "Kong"
SIZE = (400, 300)
# =====================================================================

base = Path(__file__).parent
output_dir = base / f"outputs_{DATASET}_{SIZE[0]}_{SIZE[1]}"
images_dir = base / f"images_{DATASET}_{SIZE[0]}_{SIZE[1]}"

# 自动找到所有 sparse_* 模型，选择注册图像数最多的
model_dirs = sorted(output_dir.glob("sparse_*"), key=lambda p: int(p.name.split("_")[1]))
if not model_dirs:
    raise FileNotFoundError(f"未找到稀疏模型目录: {output_dir}/sparse_*")

best_model_dir = model_dirs[0]
best_count = 0
for model_dir in model_dirs:
    recon = load_sparse_model(model_path=model_dir, images_root=images_dir)
    n = len(recon.images)
    print(f"  {model_dir.name}: {n} 张图像, {len(recon.points3D)} 个三维点")
    if n > best_count:
        best_count = n
        best_model_dir = model_dir

print(f"可视化: {best_model_dir.name} ({best_count} 张图像)")
recon = load_sparse_model(model_path=best_model_dir, images_root=images_dir)

visualize_reconstruction(
    recon.cameras,
    recon.images,
    recon.points3D,
    recon.images_root,
    filter_output=False,
)
