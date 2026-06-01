"""SfM 重建管线：预处理 → 特征提取 → 匹配 → 增量式重建 → 全局优化 → 导出"""
import cv2
import pycolmap
from pathlib import Path

# ==================== 可自定义 ====================
DATASET = "Kong"
SIZE = (400, 300)
# =================================================

input_dir = Path(f"images_{DATASET}")
image_dir = Path(f"images_{DATASET}_{SIZE[0]}_{SIZE[1]}")
output_path = Path(f"outputs_{DATASET}_{SIZE[0]}_{SIZE[1]}")

# ========== 1. 预处理：CLAHE 增强 + 下采样 ==========
image_dir.mkdir(exist_ok=True)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
for img_path in sorted(input_dir.iterdir()):
    if img_path.suffix.lower() not in extensions:
        continue
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"跳过 {img_path.name}")
        continue

    # 下采样
    resized = cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA)

    # CLAHE 增强对比度（提高 SIFT 在弱纹理区域的特征检出）
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    out_path = image_dir / img_path.name
    cv2.imwrite(str(out_path), enhanced)
    print(f"已预处理: {img_path.name} ({SIZE[0]}x{SIZE[1]})")

# ========== 2. SfM 重建 ==========
output_path.mkdir(exist_ok=True)
database_path = output_path / "database.db"
if database_path.exists():
    database_path.unlink()

# 特征提取
pixel_count = SIZE[0] * SIZE[1]
if pixel_count < 20000:
    num_features = 3000
    peak_threshold = 0.001
    edge_threshold = 15.0
elif pixel_count < 80000:
    num_features = 8000
    peak_threshold = 0.003
    edge_threshold = 10.0
else:
    num_features = 25000         # 提高特征点数
    peak_threshold = 0.004       # 略降阈值获取更多特征
    edge_threshold = 10.0

feature_options = pycolmap.FeatureExtractionOptions()
feature_options.sift = pycolmap.SiftExtractionOptions()
feature_options.sift.max_num_features = num_features
feature_options.sift.peak_threshold = peak_threshold
feature_options.sift.edge_threshold = edge_threshold
feature_options.sift.first_octave = 0            # 从原分辨率开始，不放大（图已够小）
feature_options.num_threads = 8

# 转台拍摄：所有图共享相机内参，SIMPLE_RADIAL 最稳定
reader_options = pycolmap.ImageReaderOptions()
reader_options.camera_model = "SIMPLE_RADIAL"
reader_options.default_focal_length_factor = 1.2

print(f"Step 1: 提取特征 (max_features={num_features})...")
pycolmap.extract_features(
    database_path=database_path,
    image_path=image_dir,
    image_names=[],
    camera_mode=pycolmap.CameraMode.SINGLE,
    reader_options=reader_options,
    extraction_options=feature_options
)

# 特征匹配 — 转台序列用 sequential 匹配（相邻帧才需要匹配）
print("Step 2: 特征匹配...")
sift_match_opts = pycolmap.SiftMatchingOptions()
sift_match_opts.max_ratio = 0.8                    # 收紧 ratio，减少错误匹配
sift_match_opts.cross_check = True                 # 交叉验证，提高匹配质量
sift_match_opts.max_distance = 0.7

matching_options = pycolmap.FeatureMatchingOptions()
matching_options.sift = sift_match_opts

# Sequential 匹配：相邻 N 帧互相匹配，适合转台/视频序列
sequential_opts = pycolmap.SequentialPairingOptions()
sequential_opts.overlap = 10                       # 每帧与前后各 10 帧匹配

pycolmap.match_sequential(
    database_path=database_path,
    matching_options=matching_options,
    pairing_options=sequential_opts
)

# 增量式重建
print("Step 3: 增量式重建...")

pipeline_options = pycolmap.IncrementalPipelineOptions()
pipeline_options.min_model_size = 3                # 至少 3 张图才构成模型
pipeline_options.multiple_models = True
pipeline_options.max_num_models = 5
pipeline_options.ba_refine_focal_length = True
pipeline_options.ba_refine_extra_params = True      # 优化畸变参数
pipeline_options.extract_colors = True

mapper = pipeline_options.mapper
mapper.init_min_num_inliers = 12                   # 提高初始化质量
mapper.init_max_error = 8.0                        # 收紧初始化误差
mapper.init_max_reg_trials = 5
mapper.init_min_tri_angle = 1.0                    # 提高三角化角度要求

mapper.abs_pose_min_num_inliers = 10               # 提高注册质量门槛
mapper.abs_pose_min_inlier_ratio = 0.15
mapper.abs_pose_max_error = 12.0                   # 收紧重投影误差

mapper.filter_max_reproj_error = 4.0               # 收紧滤波
mapper.filter_min_tri_angle = 1.0                  # 提高三角化角度，减少噪声点

# 局部/全局 BA（在 Pipeline 上设置，不是 mapper）
pipeline_options.ba_local_max_num_iterations = 10
pipeline_options.ba_global_max_num_iterations = 30

maps = pycolmap.incremental_mapping(
    database_path=database_path,
    image_path=image_dir,
    output_path=output_path,
    options=pipeline_options
)

if not maps:
    print("❌ 重建失败")
    print(f"   建议：检查 {input_dir} 是否有足够的重叠图片")
    exit(1)

# ========== 3. 全局 BA 精修 ==========
print("Step 4: 全局 Bundle Adjustment 精修...")
best_recon = max(maps.values(), key=lambda r: r.num_reg_images())
print(f"   最佳模型: {best_recon.num_reg_images()} 张图, 精修前 {best_recon.num_points3D()} 点")

ba_opts = pycolmap.BundleAdjustmentOptions()
ba_opts.refine_focal_length = True
ba_opts.refine_extra_params = True
ba_opts.ceres.loss_function_type = pycolmap.LossFunctionType.SOFT_L1
ba_opts.ceres.solver_options.max_num_iterations = 100

pycolmap.bundle_adjustment(best_recon, ba_opts)
print(f"   精修后: {best_recon.num_reg_images()} 张图, {best_recon.num_points3D()} 点")

# ========== 4. 导出 ==========
for rec_id, reconstruction in maps.items():
    model_dir = output_path / f"sparse_{rec_id}"
    model_dir.mkdir(exist_ok=True)
    reconstruction.write(model_dir)
    ply_path = output_path / f"pointcloud_{rec_id}.ply"
    reconstruction.export_PLY(ply_path)
    print(f"模型 {rec_id}: {reconstruction.num_reg_images()} 张图, {reconstruction.num_points3D()} 点 → {ply_path}")

total_reg = sum(r.num_reg_images() for r in maps.values())
total_pts = sum(r.num_points3D() for r in maps.values())
print(f"总计: {len(maps)} 个模型, {total_reg} 张图, {total_pts} 个三维点")
