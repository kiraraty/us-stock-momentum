"""风控层：在策略权重上叠加保险
1. VIX 切现金：VIX > 阈值时整体仓位 → 现金
2. 组合回撤减仓：实时回撤 > 阈值时砍半
"""
import pandas as pd
from data import DATA_DIR


def load_vix() -> pd.Series:
    df = pd.read_parquet(DATA_DIR / "vix.parquet")
    return df["VIX"]


def vix_filter(weights: pd.DataFrame, vix_threshold: float = 35.0) -> pd.DataFrame:
    """VIX > 阈值时全部切现金（权重 = 0）"""
    vix = load_vix()
    vix = vix.reindex(weights.index).ffill()
    mask = (vix < vix_threshold).astype(float)  # 1=持仓, 0=现金
    return weights.mul(mask, axis=0)


def drawdown_throttle(weights: pd.DataFrame, prices: pd.DataFrame,
                      dd_threshold: float = 0.15, throttle: float = 0.5) -> pd.DataFrame:
    """组合回撤超阈值 → 仓位 × throttle
    注意：用 T-1 之前的历史回撤判断，T 日生效（无 lookahead）
    """
    rets = prices.pct_change().fillna(0)
    w_shift = weights.shift(1).fillna(0)
    port_ret = (w_shift * rets).sum(axis=1)
    equity = (1 + port_ret).cumprod()
    dd = equity / equity.cummax() - 1
    # T-1 时点的回撤决定 T 日是否减仓
    in_dd = (dd.shift(1) < -dd_threshold).astype(float)
    scale = 1.0 - in_dd * (1.0 - throttle)
    return weights.mul(scale, axis=0)


def apply_risk(weights: pd.DataFrame, prices: pd.DataFrame,
               vix_threshold: float = 35.0,
               dd_threshold: float = 0.15,
               throttle: float = 0.5) -> pd.DataFrame:
    """组合应用：VIX 切现金 + DD 减仓"""
    w = vix_filter(weights, vix_threshold)
    w = drawdown_throttle(w, prices, dd_threshold, throttle)
    return w


def absolute_momentum_filter(weights: pd.DataFrame, benchmark: pd.Series,
                             lookback: int = 252) -> pd.DataFrame:
    """绝对动量过滤：benchmark 过去 lookback 日收益 < 0 时全部切现金
    用 T-1 信号决定 T 日仓位，无 lookahead
    """
    bench_ret = benchmark.pct_change(lookback)
    # T-1 日的信号决定 T 日持仓
    signal = (bench_ret.shift(1) > 0).astype(float)
    signal = signal.reindex(weights.index).fillna(0)
    return weights.mul(signal, axis=0)
