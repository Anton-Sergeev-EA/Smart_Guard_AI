# Smart_Guard_AI

[Русский](README.md) · [English](README.en.md) · **中文** · [हिन्दी](README.hi.md) · [Español](README.es.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Italiano](README.it.md)

通过序列号连接制造质量检测与状态监测的实验室概念验证。补充作品集项目，目前不扩展功能。图像和遥测均为合成数据；工业部署和商业收益未得到验证。

![Smart_Guard_AI](docs/screenshot_dashboard.png)

## 架构

PyTorch CNN 分类程序生成的表面缺陷并计算 Quality Score。LSTM 原型使用遥测与质量相关输入。C++17 可执行程序处理遥测；FastAPI 提供数字档案与 HTML/CSS/JS 仪表板。解释采用本地检索加模板，不调用外部 LLM。

## 启动与验证

从仓库根目录执行命令。Python 依赖未完全固定。训练与车队生成会覆盖生成文件；需保留现有模型时使用独立副本。仪表板：http://localhost:8000。

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cmake -S backend/cpp -B backend/cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build backend/cpp/build -j2
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8000
```

## 可选重新生成

```sh
cd backend/ml
../../.venv/bin/python generate_dataset.py
../../.venv/bin/python train_defect_model.py
../../.venv/bin/python train_predictive_model.py
../../.venv/bin/python simulate_fleet.py
```

## 证据边界

Quality Score 与未来故障的关系是研究假设。准确率、停机减少、索赔减少和经济收益需独立真实数据评估。演示地点名称不能证明安装。lead_time_days_vs_classic 是阈值日期减固定第 90 天，不是模型首次顺序告警时间。种子使用 SHA-256 代替 Python hash()；未重新训练与评估时，旧权重和指标不能归属于新数据集。C++ --bench 仅测内存计算循环，不测采集到仪表板吞吐量或工业边缘部署。

## 文档

这些是简明本地化指南；详细示例见俄文 README。界面与解释语言不变。正常输入 smoke 与图像可复现测试不验证 CNN/LSTM 准确率、RUL 或生产恢复。

[Русский](README.md)
