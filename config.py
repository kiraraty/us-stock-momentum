"""全局配置"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data_cache"
DATA_DIR.mkdir(exist_ok=True)

# ===== Alpaca (paper) =====
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_PAPER = True  # 纸面交易

# ===== 回测 & 策略参数 =====
START_DATE = "2000-01-03"
END_DATE = None  # None = 到今天
BENCHMARK = "SPY"

# 额外股票池（S&P 500 之外的知名标的）
EXTRA_TICKERS = [
    # 主流 ETF
    "QQQ", "IWM", "DIA", "VOO", "VTI", "XLK", "XLF", "XLE", "XLV", "ARKK",
    # 中概股
    "BABA", "PDD", "JD", "BIDU", "NIO", "LI", "XPEV", "TME", "BILI",
    "NTES", "TCOM", "YUMC", "ZTO", "BEKE", "IQ", "HTHT",
    # 外国龙头 ADR
    "TSM", "ASML", "NVO", "SHOP", "SAP", "TM", "HMC", "SONY", "UL",
    "RIO", "BHP", "BP", "SHEL", "AZN", "GSK", "NVS",
    # 美股新贵 / 热门非 S&P500
    "ARM", "RDDT", "COIN", "HOOD", "RBLX", "U", "SNOW", "PLTR", "NET",
    "DDOG", "CRWD", "ZS", "MDB", "OKTA", "TEAM", "DOCU", "ZM",
    "LCID", "RIVN", "FSLY", "AFRM", "SOFI", "UPST", "DKNG",
    # 加密 / 区块链相关
    "MSTR", "MARA", "RIOT", "CLSK",
]

# ========== 动量策略参数（生产配置，目标真实 30% CAGR）==========
# 来源：Optuna 在 [5, 15] top_n 区间的最优解，walk-forward 三段验证通过
# 回测（15bp 滑点）：CAGR 44.84% / Sharpe 1.73 / MaxDD -21.92%
# 真实预期（扣幸存者偏差）：CAGR ~30% / Sharpe ~1.3 / MaxDD ~-30%
LOOKBACK_DAYS = 119          # 动量回看窗口（约 6 个月）
TOP_N = 10                   # 持仓股票数
REBALANCE_FREQ = "W-FRI"     # 周五调仓
MIN_PRICE = 5.0              # 过滤低价股（<$5）
MAX_WEIGHT = 0.15            # 单票上限

# Vol targeting 参数
TARGET_VOL = 0.22            # 目标年化波动 22%
VOL_LOOKBACK = 41            # 实现波动计算窗口（约 2 个月）

# 绝对动量过滤（regime switching）
# SPY 过去 ABS_MOM_LOOKBACK 日收益 < 0 时全部切现金
# 26 年回测验证：CAGR 20.64% → 22.75%, MaxDD -67.75% → -24.78%
ABS_MOM_ENABLED = True
ABS_MOM_LOOKBACK = 252       # 12 月

# 成本假设
COMMISSION = 0.0             # Alpaca 免佣
SLIPPAGE_BPS = 15            # 15 基点滑点（包含价差 + 冲击成本，更贴近实盘）
