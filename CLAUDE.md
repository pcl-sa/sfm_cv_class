# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Structure from Motion (SfM) 3D 重建实验项目。从多视角二维图片中恢复三维点云，使用 COLMAP (pycolmap) 的增量式重建管线。

## 入口脚本

- `python script.py` — 完整管线：下采样 → SfM 重建 → 导出
- `python visualize.py` — 使用 colmap-rerun 可视化稀疏重建结果

两个脚本顶部均有 `DATASET` 和 `SIZE` 变量。`SIZE` 可选 `(400, 300)` 或 `(100, 75)`，修改后两者自动使用对应的输入/输出目录，互不干扰。`DATASET` 用于切换不同物体（数据放在 `images_{DATASET}/`），不同数据集产出完全隔离。

## 环境

- Python 环境管理: Conda
- 核心依赖: `opencv-python`, `pycolmap`, `colmap-rerun`
- 安装: `pip install -r requirements.txt`

## 工作流程

`script.py` 执行三步管线：

1. **下采样** — 将 `images_{DATASET}/` 缩放为 `SIZE` 指定尺寸，输出到 `images_{DATASET}_{w}_{h}/`
2. **SfM 重建** — SIFT 特征提取（自适应特征点数）→ 穷举匹配 → 增量式重建
3. **导出** — 稀疏模型写入 `outputs_{DATASET}_{w}_{h}/sparse_{id}/`，PLY 点云写入 `outputs_{DATASET}_{w}_{h}/pointcloud_{id}.ply`

特征/匹配/重建参数均根据分辨率自适应调整（见代码中的 `pixel_count` 判断分支）。
低分辨率下自动降低特征阈值、放宽匹配 ratio、降低注册内点数要求。
`multiple_models=True` 允许 COLMAP 从不同初始对重试，减少 "Could not register"。

## 目录结构

- `images_{DATASET}/` — 原始输入图片，如 `images_Kong/`、`images_south_build/`
- `images_{DATASET}_{w}_{h}/` — 下采样后的图片（自动生成，SfM 输入）
- `outputs_{DATASET}_{w}_{h}/` — SfM 重建结果：`database.db`、`sparse_{id}/`、`pointcloud_{id}.ply`
