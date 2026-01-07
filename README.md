# Wafer Thickness Viewer (Streamlit)

一个用于晶圆厚度数据的 2D 顶视图 + 3D 曲面可视化工具，支持：
- CSV 导入
- 坐标生成器（十字/同心圆/网格）
- 数据集管理（多片 wafer 命名切换）
- 统计指标
- 3D 视角控制
- Spec 上下限可视化

## CSV 格式
必须包含列：`x`, `y`, `thickness`（大小写不敏感）
- x/y 单位：mm
- 原点：晶圆中心 (0,0)

## 本地运行
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud 部署
1. 将本仓库推送到 GitHub
2. 打开 Streamlit Community Cloud -> New app
3. 选择该 repo / 分支
4. Main file path 填：`app.py`
5. Deploy
