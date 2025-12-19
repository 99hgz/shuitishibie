# 水体识别评估流水线

该仓库提供脚本 `src/water_evaluation.py`，用于结合 Sentinel-2 卫星影像、GSW occurrence 数据、glm-4.6v VLM 与 SAM 分割模型，对中国区域 100 个随机样本的水体识别性能进行评估，并生成 HTML 报告。

## 环境准备
1. 安装依赖：
```bash
pip install -r requirements.txt
```
2. 确保本地已配置 Google Earth Engine 认证，并在需要时提供 `GEE_PROJECT`。
3. 准备以下环境变量：
- `OPENAI_API_KEY`：访问 glm-4.6v 模型的密钥。
- `SAM_CHECKPOINT`：SAM 模型权重路径（默认 `sam_vit_h.pth`）。

## 运行
```bash
python src/water_evaluation.py
```
脚本将：
1. 在中国境内随机采样 100 个区域，从 Sentinel-2 获取 RGB 影像和对应的 GSW occurrence 栅格。
2. 将影像传给 glm-4.6v，判断是否存在水体并返回 bounding box。
3. 使用 SAM 按 bbox 分割水体。
4. 将分割结果与 GSW occurrence 对比，计算 Precision/Recall/IoU。
5. 生成 `reports/report.html`，展示输入影像、occurrence 真实值、预测叠加以及每块区域的指标汇总。

如需调整采样数量、时间范围或云量阈值，可在 `TileConfig` 中修改相关参数。
