"""使用 colmap-rerun 可视化 SfM 稀疏重建结果"""
from pathlib import Path
from colmap_rerun.core.reconstruction import load_sparse_model
from colmap_rerun.visualization.visualizer import visualize_reconstruction

# ==================== 可自定义：与 script.py 保持一致 ====================
DATASET = "Kong"
SIZE = (1200, 900)
# SIZE = (100, 75)
# =====================================================================

base = Path(__file__).parent

recon = load_sparse_model(
    model_path=base / f"outputs_{DATASET}_{SIZE[0]}_{SIZE[1]}" / "sparse_0",
    images_root=base / f"images_{DATASET}_{SIZE[0]}_{SIZE[1]}",
)

visualize_reconstruction(
    recon.cameras,
    recon.images,
    recon.points3D,
    recon.images_root,
    filter_output=False,
)
