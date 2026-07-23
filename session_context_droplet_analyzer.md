# Image Identification Session Context — DropletAnalyzer Project

## 项目概述

项目名称：DropletAnalyzer（液滴分析仪）
项目路径：`C:\LYD\imageidentification`
目标：基于 OpenCV 的液滴图像检测与分析桌面应用（PyQt6 界面）
当前版本：v2（时域中值背景减除 + CLAHE + 条纹去除 + GT 参数调优）
入口文件：`main.py`（主入口）、`app/main_app.py`（核心应用，1639 行）

## 用户偏好与指令

### 关键指令（必须遵守）

1. **版本管理**：修改软件时保留原版，新版本在后面加版本号。
   - 用户原话："从现在开始，对软件进行修改时要保留原来的软件，修改的软件在后面添加版本号"（2026-07-23）
   - 做法：旧版 EXE 重命名为 `DropletAnalyzer_v1.exe`，新版输出为 `DropletAnalyzer.exe`

2. **临时文件位置**：所有临时文件必须存放在工作目录下的 `output/temporary/` 文件夹，不得放在用户目录下。
   - 用户原话："临时文件存放在工作目录下单独的temporary文件夹，不要放在用户目录下"（2026-07-22）

3. **交付物格式**：必须是单个独立应用程序，不要任何脚本、启动程序或依赖。
   - 用户原话："不要任何脚本和启动程序以及依赖，我只要一个应用程序本体"（2026-07-22）
   - 做法：PyInstaller `--onefile` 打包为单个 `.exe`

4. **视觉标记与统计一致**：`scale_factor` 必须为 1.0，确保标记面积 = 实际计算面积。
   - 用户原话："图上自动识别产生的标记不能够反应实际计算的像素面积，根据实际计算的像素面积进行标记"（2026-07-22）

5. **Add 模式**：十字光标放到液滴内部时自动识别液滴边界，不是手动绘制多边形。
   - 用户原话："add添加的逻辑也有极大的问题，完全没有实现前述的将十字光标放到液滴内部自动识别液滴的功能"（2026-07-23）

### 核心偏好

- 使用**时域中值背景减除**（temporal median background subtraction），用户认为中值处理后液滴白色、背景黑色，对比度更高
- 用户原话："经过背景处理后的黑白照片难道不是更容易识别吗，现在液滴为白色，背景为黑色"（2026-07-23）
- 通过**浏览器端参数调优器**（`tuner.html` via `tuner_server.py`，端口 8080）手动调参后应用到应用
- 偏好**手动输入坐标 + 鼠标操作并存**的交互方式
- 要求根据**人工标注（ground truth）**进行参数网格搜索与 F1 优化
- 检测算法需针对中值处理后的高对比度图像（液滴白色、背景黑色）进行优化

## 当前技术架构

### 检测引擎 (`detect_droplets` 函数)

中值背景路径（推荐，`use_temporal=True`）：
```
多帧中值背景构建 → 中值相减(residual) → 列条纹去除 → CLAHE增强 → 二元阈值 → 形态学开/闭运算 → 距离变换 → 分水岭 → 面积/长宽比过滤
```

高斯路径（回退，`use_temporal=False`）：
```
高斯模糊背景估计 → 归一化 → Otsu阈值 → 峰值检测 → 分水岭
```

### GT 调优后的最优参数（中值路径）

| 参数 | 值 | 含义 |
|------|------|------|
| T | 5 | 二元阈值（残差 > 5 为前景） |
| MD | 8 | 液滴间最小距离（像素） |
| MA | 6 | 最小面积（像素，≈264 μm D_eq） |
| MO | 1 | 形态学开运算迭代次数 |
| PT | 0.04 | 峰值阈值（距离变换最大值的 4%） |
| SF | 1.0 | 视觉缩放（=1.0，标记面积 = 统计面积） |

### 预处理管线（四条路径统一）

中值背景减除 → 列条纹去除（滑动窗口移动平均） → CLAHE 增强（clipLimit=3.0, tileGridSize=16×16）

四条路径使用完全相同的处理流程：
1. 检测引擎（`detect_droplets`）
2. 预处理预览（`_pre_apply`）
3. 保存预处理图像（`_save_preprocessed`）
4. 跳过保存直接检测（`_skip_to_detect`）

### 中值背景构建 (`_build_median_bg`)

- 从当前图像所在文件夹读取**全部**图像（非仅前 150 帧）
- 对每帧进行中值模糊（3px）→ 列条纹去除 → 逐像素取中值
- 缓存到 `_cached_median_bg`

### 应用界面（4 个标签页）

1. **Crop Images** — 文件夹批量裁剪，支持手动坐标输入和重用上次裁剪
2. **Preprocess**（新增）— 背景校正 + CLAHE，支持预览、保存（可选）、跳过保存直接检测
3. **Detect & Review** — 运行检测、浏览结果、Add/Delete 手动修正
4. **Export Results** — 生成 Excel + 图表 + 标注图像

## 已知问题与修复记录（历史回顾）

### PyQt6 兼容性问题

1. **`event.pos()` 不存在** — PyQt6 中 `QEvent` 没有 `pos()` 方法，必须用 `event.position().toPoint()`
   - 涉及文件：`app/main_app.py` 的 `eventFilter`、`_on_press`、`_on_move`、`_on_release`
   - 所有位置均已替换

2. **`eventFilter` 返回类型** — 必须返回 `bool`（`True`/`False`），不能返回 `None`
   - 修复：在每个 `eventFilter` 分支末尾显式 `return True` 或 `return False`

3. **`crash_log.txt` 写入失败** — 打包后 `output/` 子文件夹不存在导致写入报错
   - 修复：删除所有 `crash_log.txt` 写入代码

### OpenCV 兼容性问题

4. **`cv2.GaussianBlur` 对 1D 数组崩溃** — `(h=1, w)` 形状输入时失败
   - 修复：替换为纯 numpy 滑动窗口移动平均（每个像素列减去相邻 50 列的均值）

### 检测算法历史问题

5. **时域中值后液滴检测全部错误** — 原因是归一化步骤 `residual / median_bg` 破坏了信号
   - 修复：跳过归一化，直接对残差应用 CLAHE

6. **Add 模式闪退** — `_auto_detect_at` 未使用中值背景
   - 修复：在 `_run_det_one` 中缓存中值背景到 `_cached_median_bg`，Add 模式下使用该值

7. **Tuner 参数不生效** — 前端 JS 发送的参数名与后端 Python 接收的参数名不匹配
   - 修复：统一前后端参数名

## 关键文件清单

| 文件 | 说明 |
|------|------|
| `main.py` | 应用入口点 |
| `app/main_app.py`（1639 行）| 主应用，包含所有 UI 和检测逻辑 |
| `tuner_server.py` | HTTP 参数调优服务器（localhost:8080/tuner.html） |
| `tuner.html` | 参数调优浏览器界面 |
| `DropletAnalyzer.spec` | PyInstaller 打包配置 |
| `dist/DropletAnalyzer.exe` | v2 构建产物（245MB，2026-07-23） |
| `dist/DropletAnalyzer_v1.exe` | v1 备份（高斯法） |

## 已验证的数据集

| 数据集 | 类型 | 检测结果 |
|--------|------|---------|
| H0 | 暗液滴 | 552 droplets |
| H1 | 亮液滴 | 459 droplets |
| H0_v2 | 混合 | 389 droplets |

## PyInstaller 打包注意事项

- 使用 `--onefile` 模式
- 需要 `--collect-binaries PyQt6` 确保 Qt 平台插件（qwindows.dll）被打包
- `main.py` 中需设置 `QT_QPA_PLATFORM_PLUGIN_PATH` 环境变量
- 历史上 `--windowed` 模式会隐藏错误信息，建议先以 `--console` 模式确认可运行后再考虑隐藏控制台
- 用户明确不需要任何 `.bat`、`.vbs` 脚本或单独的 `_internal` 文件夹

## 用户沟通风格

- 使用中文交流
- 直接指出问题，期望快速修复
- 偏好通过浏览器（tuner）可视化验证后确认参数
- 当修复不完整时会反复提出同一需求直到完全实现
- 关注最终交付物的简洁性（单个 EXE，无依赖）
