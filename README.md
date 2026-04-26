# US Stock Quant (S&P 500 动量策略 + Alpaca 纸面交易)

## 结构
- `config.py` – 参数和 API key
- `data.py` – yfinance 拉 S&P 500 + 历史行情（parquet 缓存）
- `strategy.py` – 横截面动量：过去 126 日收益 top 20，周频调仓
- `backtest.py` – 向量化回测 + 夏普/回撤/CAGR/胜率
- `broker.py` – Alpaca 纸面交易封装
- `live.py` – 实盘入口：算目标权重 → 调仓

## 快速开始
```bash
pip install -r requirements.txt

# 1. 回测（第一次会下载 ~500 只股票历史，耐心等几分钟）
python backtest.py

# 2. 纸面交易 dry run（不真下单）
python live.py --dry

# 3. 真的下单到 Alpaca paper
python live.py
```

## 策略思路
横截面动量（cross-sectional momentum）：
1. 每周五按过去 6 个月收益对 S&P 500 排序
2. 等权买入前 20 只，单票上限 10%
3. 周五调仓，其他日子持有不动
4. 滑点 5bp，免佣

这是最经典、最鲁棒的因子之一（Jegadeesh & Titman 1993）。
跑通后可以扩展：反转、低波、质量、多因子组合。

## 后续可扩展
- 加止损 / 波动目标（vol targeting）
- 多策略组合（动量 + 均值回归）
- 用 Alpaca 实时行情流替代 yfinance（更低延迟）
- 定时任务：macOS launchd 或 crontab
