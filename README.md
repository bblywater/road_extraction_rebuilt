# Road Extraction Rebuilt

重建目标：

- Massachusetts Roads 与 DeepGlobe 数据加载
- Baseline U-Net、Optimized U-Net、DLGU-Net
- 训练、评估、断点恢复、单图预测
- 指标统计、图表、Markdown/JSON/CSV 报告
- 数据集探索与对比可视化

统一目录结构：

- `data/massachusetts/train/data`
- `data/massachusetts/train/label`
- `data/massachusetts/val/data`
- `data/massachusetts/val/label`
- `data/massachusetts/test/data`
- `data/massachusetts/test/label`

典型命令：

```bash
python scripts/prepare_massachusetts.py
python scripts/train.py --config configs/optimized_unet_massachusetts.yaml
python scripts/evaluate.py --experiment experiments/optimized_unet_massachusetts
python scripts/predict.py --experiment experiments/optimized_unet_massachusetts --image data/massachusetts/test/data/10378780_15.tiff
```
