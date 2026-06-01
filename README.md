# Structure from Motion (SfM) 三维重建

从多视角二维图片中恢复三维稀疏点云，基于 [COLMAP](https://colmap.github.io/)（pycolmap）增量式重建管线。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备图片：放入 images_{物体名}/ 目录（如 images_Kong/）

# 3. 修改 script.py 顶部的 DATASET 和 SIZE，然后运行
python script.py

# 4. 可视化重建结果
python visualize.py
```

## 配置

`script.py` 和 `visualize.py` 顶部有两个配置变量：

| 变量 | 说明 | 示例 |
|------|------|------|
| `DATASET` | 数据集名称，对应 `images_{DATASET}/` 目录 | `"Kong"`, `"south_build"` |
| `SIZE` | 重建分辨率（宽, 高） | `(1200, 900)`, `(400, 300)` |

修改后两个脚本同步保持一致即可，不同配置的输入/输出**完全隔离**。

## 管线流程

```
原始图片 (images_{DATASET}/)
    │  下采样 cv2.resize → SIZE
    ▼
缩放图片 (images_{DATASET}_{w}_{h}/)
    │  SIFT 特征提取 → 穷举匹配 → 增量式重建
    ▼
稀疏模型 (outputs_{DATASET}_{w}_{h}/sparse_{id}/)
    │  导出 PLY
    ▼
点云文件 (outputs_{DATASET}_{w}_{h}/pointcloud_{id}.ply)
```

参数根据分辨率自动调整：低分辨率自动降低特征阈值、放宽匹配 ratio、降低注册内点数要求。重建失败时会自动从不同初始对重试（`multiple_models=True`）。

## 目录结构

```
.
├── script.py              # 主管线入口
├── visualize.py           # 3D 可视化入口
├── requirements.txt       # Python 依赖
├── images_{DATASET}/             # 原始输入图片（需自行准备）
├── images_{DATASET}_{w}_{h}/     # 下采样图片（自动生成）
└── outputs_{DATASET}_{w}_{h}/    # 重建结果（自动生成）
    ├── database.db               # 特征/匹配数据库
    ├── sparse_0/                 # 稀疏模型（cameras.bin, images.bin, points3D.bin）
    └── pointcloud_0.ply          # PLY 点云（可用 MeshLab 打开）
```

## 依赖

- **opencv-python** — 图像缩放
- **pycolmap** — COLMAP 完整 SfM 管线（SIFT 特征提取、特征匹配、增量式重建）
- **colmap-rerun** — 基于 rerun.io 的 3D 点云可视化

## 常见问题

### 大量 "Could not register"

1. **提高分辨率** — 将 `SIZE` 改为 `(1200, 900)` 或更大
2. **检查图片重叠** — 确保相邻图片之间有足够的重叠区域（>60%）
3. **减少图片数量** — 如果一次性注册太少，尝试减少图片张数
4. **拍摄建议** — 环绕物体拍摄，每 10-20° 一张，保持连续

### Windows 下 Conda 环境

推荐使用 Conda 管理 Python 环境，项目 `.vscode/settings.json` 已配置默认使用 Conda。
