# US Stock Momentum Strategy

美股个股动量轮动策略，基于 S&P 500 成分股，26 年全量回测验证。

## 策略概览

横截面动量（Cross-sectional Momentum）：买入近期涨幅最强的股票，持有并定期轮动。

**核心逻辑**：动量因子是学术界最稳健的 alpha 来源之一（Jegadeesh & Titman 1993），核心假设是"涨的继续涨"。配合波动率目标化和宏观 regime 过滤，实现风险可控的绝对收益。

## 支持策略

| 策略 | 说明 |
|------|------|
| `momentum` | 纯价格动量，等权持有 top N |
| `momentum_voltarget` | **推荐** 波动率目标化，年化波动锚定 22% |
| `quality` | 质量因子（ROE、营收增速） |
| `momentum_12_1` | 12-1 月动量（经典 Jegadeesh-Titman） |
| `sector_neutral` | 行业市值中性动量 |
| `risk_parity` | 风险平价 |

## 回测结果

### 26 年全量（2000-2026）

| 策略 | 年化 | 夏普 | 最大回撤 |
|------|------|------|---------|
| momentum_12_1 | 29.1% | 0.98 | -67.5% |
| quality | 23.6% | 0.96 | -56.6% |
| **momentum_voltarget** | **19.5%** | **0.92** | **-65.3%** |
| momentum | 24.6% | 0.89 | -70.7% |
| SPY | 8.1% | 0.50 | -55.2% |

注：26 年包含 dot-com 泡沫和金融危机，策略表现被极端行情拉低。

### 2010-2026 近 17 年（实盘最参考区间）

| 策略 | 年化 | 夏普 | 最大回撤 | 跑赢 SPY |
|------|------|------|---------|---------|
| **voltarget** | **29.9%** | **1.47** | **-26.3%** | **14/17 年** |
| 12_1 | 42.5% | 1.09 | -40.7% | 15/17 |
| quality | 32.2% | 1.16 | -33.4% | 14/17 |
| momentum | 40.4% | 1.36 | -39.4% | 15/17 |
| SPY | 13.9% | 1.05 | -33.7% | 0/17 |

### 关键结论

1. **momentum_voltarget 最均衡**：年化 30%、夏普 1.47、最大回撤 -26.3%
2. **波动率目标化有效**：将回撤从 -70% 压到 -26%（2010-2026 区间）
3. **熊市保护显著**：2022 年 SPY -18.6%，voltarget 仅 +2.0%
4. **真实预期**：年化 20-25%、夏普 1.0-1.3、回撤 -20% 以内

## 策略详解：voltarget（推荐实盘）

### 核心逻辑

```
目标仓位 = 22% / 实现波动率
```

- 当某只股票波动率低 → 放大仓位
- 当某只股票波动率高 → 缩小仓位
- 组合整体波动率锚定在 22% 年化

### 关键参数

```python
LOOKBACK_DAYS = 119        # 动量回看窗口（~6 个月）
TOP_N = 10                 # 持仓股票数
REBALANCE_FREQ = "W-FRI"  # 周五调仓
TARGET_VOL = 0.22          # 目标年化波动率 22%
ABS_MOM_ENABLED = True     # SPY 12个月均线以下切现金
SLIPPAGE_BPS = 15          # 15bp 滑点
```

## 文件结构

```
us-stock/
├── config.py              # 全局配置（参数、API keys）
├── data.py                # yfinance 数据拉取和缓存
├── strategy.py            # 策略定义（多策略支持）
├── backtest.py            # 回测引擎（向量化）
├── optimize.py            # Optuna 参数优化
├── walk_forward.py        # Walk-forward 验证
├── stress_test.py         # 压力测试
├── live.py                # 实盘入口（Alpaca paper）
├── broker.py              # Alpaca API 封装
├── compare_strategies.py  # 多策略横向对比
├── run_all_strategies.py  # 全策略批量回测
├── docs/
│   ├── why_us_momentum_works_crypto_doesnt.md
│   └── etf_rotation_lesson.md
└── data_cache/            # parquet 缓存（不提交）
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 Alpaca API（paper trading）
cp .env.example .env
# 填入 ALPACA_API_KEY 和 ALPACA_SECRET_KEY

# 3. 回测
python backtest.py

# 4. 模拟盘 dry run（不真下单）
python live.py --dry

# 5. 真下单到 Alpaca paper
python live.py
```

## 数据说明

- 数据源：Yahoo Finance，S&P 500 成分股
- 时间范围：2000-01-03 ～ 今日
- 缓存：parquet 格式，`data_cache/` 目录
- 约 500 只股票，26 年数据，首次运行自动下载

## 局限性

1. **极端行情仍有大回撤**：2002 年 -49%、2008 年 -34%，波动率目标化不能完全消除黑天鹅
2. **幸存者偏差**：回测只用当前存活的 S&P 500 成分股
3. **实盘会有差距**：滑点、流动性、执行延迟都会拉低收益
4. **过去表现不代表未来**：26 年数据不能保证未来仍然有效

## 参考资料

- Jegadeesh & Titman (1993) - Returns to Buying Winners and Selling Losers
- Asness et al. - Time Series Momentum
- Lowenstein - When Genius Failed（长期资本陨落启示）

## 免责声明

本项目仅供研究学习，不构成投资建议。实盘亏损自负。
