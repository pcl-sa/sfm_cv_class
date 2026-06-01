"""SfM 重建管线：下采样 → 特征提取 → 匹配 → 增量式重建 → 导出"""
import cv2
import pycolmap
from pathlib import Path

# ==================== 可自定义 ====================
DATASET = "Kong"        # 数据集名称，对应 images_{DATASET}/ 目录
SIZE = (1200, 900)       # 重建分辨率
# SIZE = (100, 75)      # 低分辨率备选
# =================================================

input_dir = Path(f"images_{DATASET}")
image_dir = Path(f"images_{DATASET}_{SIZE[0]}_{SIZE[1]}")
output_path = Path(f"outputs_{DATASET}_{SIZE[0]}_{SIZE[1]}")

# ========== 1. 下采样 ==========
image_dir.mkdir(exist_ok=True)

extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
for img_path in sorted(input_dir.iterdir()):
    if img_path.suffix.lower() not in extensions:
        continue
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"跳过 {img_path.name}")
        continue
    resized = cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA)
    out_path = image_dir / img_path.name
    cv2.imwrite(str(out_path), resized)
    print(f"已缩放: {img_path.name} -> {out_path.name} ({SIZE[0]}x{SIZE[1]})")

# ========== 2. SfM 重建 ==========
output_path.mkdir(exist_ok=True)
database_path = output_path / "database.db"

if database_path.exists():
    database_path.unlink()

# 特征提取 — 根据图像分辨率自适应特征点数
pixel_count = SIZE[0] * SIZE[1]
if pixel_count < 20000:
    num_features = 2000
    peak_threshold = 0.001
    edge_threshold = 15.0
elif pixel_count < 80000:
    num_features = 6000
    peak_threshold = 0.003
    edge_threshold = 10.0
else:
    num_features = 15000
    peak_threshold = 0.005
    edge_threshold = 8.0

feature_options = pycolmap.FeatureExtractionOptions()
feature_options.sift = pycolmap.SiftExtractionOptions()
feature_options.sift.max_num_features = num_features
feature_options.sift.peak_threshold = peak_threshold
feature_options.sift.edge_threshold = edge_threshold
feature_options.num_threads = 4

print(f"Step 1: 提取特征 (max_features={num_features}, peak_threshold={peak_threshold})...")
pycolmap.extract_features(
    database_path=database_path,
    image_path=image_dir,
    image_names=[],
    camera_mode=pycolmap.CameraMode.AUTO,
    extraction_options=feature_options
)

# 特征匹配 — 根据分辨率调整匹配宽松度
print("Step 2: 特征匹配...")
sift_match_opts = pycolmap.SiftMatchingOptions()
sift_match_opts.max_ratio = 0.95 if pixel_count < 20000 else 0.85
sift_match_opts.cross_check = False
sift_match_opts.max_distance = 1.0

matching_options = pycolmap.FeatureMatchingOptions()
matching_options.sift = sift_match_opts

pairing_options = pycolmap.ExhaustivePairingOptions()
pairing_options.block_size = 50

pycolmap.match_exhaustive(
    database_path=database_path,
    matching_options=matching_options,
    pairing_options=pairing_options
)

# 增量式重建 — 针对注册失败问题全面放宽参数
print("Step 3: 增量式重建...")

pipeline_options = pycolmap.IncrementalPipelineOptions()
pipeline_options.min_model_size = 2
pipeline_options.multiple_models = True
pipeline_options.max_num_models = 3
pipeline_options.ba_refine_focal_length = True
pipeline_options.extract_colors = True

mapper = pipeline_options.mapper
mapper.init_min_num_inliers = 8
mapper.init_max_error = 16.0
mapper.init_max_reg_trials = 5
mapper.init_min_tri_angle = 0.5
mapper.abs_pose_min_num_inliers = 5
mapper.abs_pose_min_inlier_ratio = 0.05
mapper.abs_pose_max_error = 24.0
mapper.filter_max_reproj_error = 8.0
mapper.filter_min_tri_angle = 0.3

maps = pycolmap.incremental_mapping(
    database_path=database_path,
    image_path=image_dir,
    output_path=output_path,
    options=pipeline_options
)

if maps:
    for rec_id, reconstruction in maps.items():
        model_dir = output_path / f"sparse_{rec_id}"
        model_dir.mkdir(exist_ok=True)
        reconstruction.write(model_dir)
        ply_path = output_path / f"pointcloud_{rec_id}.ply"
        reconstruction.export_PLY(ply_path)
        print(f"模型 {rec_id}: {reconstruction.num_reg_images()} 张图像注册, {reconstruction.num_points3D()} 个三维点")
        print(f"点云文件: {ply_path}")
    total_reg = sum(r.num_reg_images() for r in maps.values())
    print(f"总计: {len(maps)} 个模型, 共 {total_reg} 张图像成功注册")
else:
    print("❌ 重建失败，请检查图片重叠度和数量。")
    print("   建议：1) 确保图片之间有足够的重叠区域")
    print("         2) 尝试 SIZE = (400, 300) 提高分辨率")
    print(f"         3) 检查 {input_dir} 目录是否有图片")
