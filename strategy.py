"""策略库：所有策略统一输出 wide 权重表（index=date, columns=ticker, 值=权重）
- momentum      : 横截面动量（过去 N 日收益 top K）
- momentum_12_1 : 经典 12-1 动量（过去 12 月 - 最近 1 月）
- mean_reversion: 短期反转（过去 N 日跌得最多的 top K）
- low_vol       : 低波因子（过去 N 日波动率最低的 top K）
- dual_ma       : 双均线（每只票独立，快线>慢线则持有，等权）
"""
import pandas as pd
import numpy as np
from config import (
    LOOKBACK_DAYS, TOP_N, REBALANCE_FREQ, MIN_PRICE, MAX_WEIGHT,
    TARGET_VOL, VOL_LOOKBACK, ABS_MOM_ENABLED, ABS_MOM_LOOKBACK,
)


def _rebalance_dates(prices: pd.DataFrame, freq: str) -> list:
    rd = prices.resample(freq).last().index
    return [d for d in rd if d in prices.index]


def _apply_weights(prices: pd.DataFrame, scores_by_date: dict,
                   top_n: int, max_w: float) -> pd.DataFrame:
    """scores_by_date: {date: Series(score, index=ticker)}  → 权重表
    正确做法：只在调仓日产生权重行，然后 reindex 到全日期并 ffill。
    """
    rebal_rows = {}
    for d, score in scores_by_date.items():
        top = score.nlargest(top_n).index
        if len(top) == 0:
            continue
        w = min(1.0 / len(top), max_w)
        row = pd.Series(0.0, index=prices.columns)
        row[top] = w
        rebal_rows[d] = row
    if not rebal_rows:
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w_df = pd.DataFrame(rebal_rows).T.sort_index()
    return w_df.reindex(prices.index).ffill().fillna(0.0)


# ============ 策略实现 ============

def momentum(prices: pd.DataFrame, lookback: int = LOOKBACK_DAYS,
             top_n: int = TOP_N, freq: str = REBALANCE_FREQ,
             min_price: float = MIN_PRICE, max_w: float = MAX_WEIGHT) -> pd.DataFrame:
    """纯动量：过去 lookback 日累计收益"""
    mom = prices.pct_change(lookback)
    valid = prices >= min_price
    scores = {}
    for d in _rebalance_dates(prices, freq):
        scores[d] = mom.loc[d].where(valid.loc[d])
    return _apply_weights(prices, scores, top_n, max_w)


def momentum_12_1(prices: pd.DataFrame, top_n: int = TOP_N,
                  freq: str = REBALANCE_FREQ,
                  min_price: float = MIN_PRICE, max_w: float = MAX_WEIGHT) -> pd.DataFrame:
    """12-1 动量：过去 252 日收益 - 过去 21 日收益（跳过最近 1 个月以避开反转）"""
    r252 = prices.pct_change(252)
    r21 = prices.pct_change(21)
    mom = r252 - r21
    valid = prices >= min_price
    scores = {}
    for d in _rebalance_dates(prices, freq):
        scores[d] = mom.loc[d].where(valid.loc[d])
    return _apply_weights(prices, scores, top_n, max_w)


def mean_reversion(prices: pd.DataFrame, lookback: int = 5,
                   top_n: int = TOP_N, freq: str = REBALANCE_FREQ,
                   min_price: float = MIN_PRICE, max_w: float = MAX_WEIGHT) -> pd.DataFrame:
    """短期反转：过去 lookback 日跌得最多的 top N"""
    ret = prices.pct_change(lookback)
    valid = prices >= min_price
    scores = {}
    for d in _rebalance_dates(prices, freq):
        s = -ret.loc[d].where(valid.loc[d])  # 跌得多 = 分数高
        scores[d] = s
    return _apply_weights(prices, scores, top_n, max_w)


def low_vol(prices: pd.DataFrame, lookback: int = 60,
            top_n: int = TOP_N, freq: str = REBALANCE_FREQ,
            min_price: float = MIN_PRICE, max_w: float = MAX_WEIGHT) -> pd.DataFrame:
    """低波因子：过去 lookback 日收益率标准差最低的 top N"""
    daily = prices.pct_change()
    vol = daily.rolling(lookback).std()
    valid = prices >= min_price
    scores = {}
    for d in _rebalance_dates(prices, freq):
        s = -vol.loc[d].where(valid.loc[d])  # 波动低 = 分数高
        scores[d] = s
    return _apply_weights(prices, scores, top_n, max_w)


def dual_ma(prices: pd.DataFrame, fast: int = 20, slow: int = 60,
            max_w: float = MAX_WEIGHT) -> pd.DataFrame:
    """双均线：每只票快线>慢线则持有，所有持有票等权"""
    ma_f = prices.rolling(fast).mean()
    ma_s = prices.rolling(slow).mean()
    signal = (ma_f > ma_s).astype(float)
    # 每行归一化为等权
    count = signal.sum(axis=1).replace(0, 1)
    weights = signal.div(count, axis=0).clip(upper=max_w)
    return weights


def risk_parity(prices: pd.DataFrame, lookback: int = 60, top_n: int = 30,
                freq: str = REBALANCE_FREQ, min_price: float = MIN_PRICE,
                max_w: float = MAX_WEIGHT) -> pd.DataFrame:
    """风险平价：从低波的 top_n 只里按 1/vol 加权"""
    daily = prices.pct_change()
    vol = daily.rolling(lookback).std()
    valid = prices >= min_price
    rebal_rows = {}
    for d in _rebalance_dates(prices, freq):
        v = vol.loc[d].where(valid.loc[d]).dropna()
        v = v[v > 0.002]  # 过滤异常低波动
        if v.empty:
            continue
        picks = v.nsmallest(top_n)
        inv = 1.0 / picks
        w = inv / inv.sum()
        w = w.clip(upper=max_w)
        w = w / w.sum()
        row = pd.Series(0.0, index=prices.columns)
        row[w.index] = w.values
        rebal_rows[d] = row
    if not rebal_rows:
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w_df = pd.DataFrame(rebal_rows).T.sort_index()
    return w_df.reindex(prices.index).ffill().fillna(0.0)


def multi_factor(prices: pd.DataFrame, top_n: int = TOP_N,
                 freq: str = REBALANCE_FREQ, min_price: float = MIN_PRICE,
                 max_w: float = MAX_WEIGHT) -> pd.DataFrame:
    """多因子打分：动量 + 低波 + (简版)质量 的 z-score 加权"""
    mom = prices.pct_change(126)
    daily = prices.pct_change()
    vol60 = daily.rolling(60).std()
    # 简版"质量"：过去 252 日夏普（用日收益均值/标准差）
    sharpe252 = daily.rolling(252).mean() / daily.rolling(252).std()

    def zscore(row):
        return (row - row.mean()) / (row.std() + 1e-9)

    scores = {}
    valid = prices >= min_price
    for d in _rebalance_dates(prices, freq):
        v = valid.loc[d]
        z_mom = zscore(mom.loc[d].where(v))
        z_lowvol = zscore(-vol60.loc[d].where(v))      # 波动越低分越高
        z_q = zscore(sharpe252.loc[d].where(v))
        combined = (z_mom + z_lowvol + z_q) / 3
        scores[d] = combined
    return _apply_weights(prices, scores, top_n, max_w)


def sector_neutral_momentum(prices: pd.DataFrame, lookback: int = LOOKBACK_DAYS,
                            per_sector: int = 3, freq: str = REBALANCE_FREQ,
                            min_price: float = MIN_PRICE,
                            max_w: float = MAX_WEIGHT) -> pd.DataFrame:
    """行业中性动量：每个 GICS 行业取动量前 per_sector 只，等权"""
    from data import get_sector_map
    sectors = get_sector_map()
    mom = prices.pct_change(lookback)
    valid = prices >= min_price

    tickers_by_sector: dict[str, list[str]] = {}
    for t in prices.columns:
        s = sectors.get(t)
        if s:
            tickers_by_sector.setdefault(s, []).append(t)

    rebal_rows = {}
    for d in _rebalance_dates(prices, freq):
        row_score = mom.loc[d].where(valid.loc[d])
        picks = []
        for sec, tks in tickers_by_sector.items():
            s_row = row_score.reindex(tks).dropna()
            picks.extend(s_row.nlargest(per_sector).index.tolist())
        if not picks:
            continue
        w = min(1.0 / len(picks), max_w)
        row = pd.Series(0.0, index=prices.columns)
        row[picks] = w
        rebal_rows[d] = row
    if not rebal_rows:
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w_df = pd.DataFrame(rebal_rows).T.sort_index()
    return w_df.reindex(prices.index).ffill().fillna(0.0)


def vol_target(weights: pd.DataFrame, prices: pd.DataFrame,
               target_vol: float = 0.15, lookback: int = 30,
               max_leverage: float = 1.0) -> pd.DataFrame:
    """波动率目标：按最近 lookback 日组合实际波动 → 按 target/realized 缩放仓位"""
    rets = prices.pct_change().fillna(0)
    w_shift = weights.shift(1).fillna(0)
    port_ret = (w_shift * rets).sum(axis=1)
    realized = port_ret.rolling(lookback).std() * (252 ** 0.5)
    scale = (target_vol / realized).clip(upper=max_leverage).fillna(1.0)
    return weights.mul(scale, axis=0)


def _quality_score() -> pd.Series:
    """从 fundamentals 缓存计算 quality 复合分数（每只票一个数）
    分量：ROE↑ + ROA↑ + 毛利率↑ + 净利率↑ + D/E↓
    每个分量做 winsorize + zscore，再等权加总
    """
    from fundamentals import load_fundamentals
    df = load_fundamentals()

    def winsorize(s: pd.Series, q: float = 0.01) -> pd.Series:
        lo, hi = s.quantile(q), s.quantile(1 - q)
        return s.clip(lower=lo, upper=hi)

    def zscore(s: pd.Series) -> pd.Series:
        s = winsorize(s)
        return (s - s.mean()) / (s.std() + 1e-9)

    z_roe = zscore(df["returnOnEquity"])
    z_roa = zscore(df["returnOnAssets"])
    z_gm = zscore(df["grossMargins"])
    z_pm = zscore(df["profitMargins"])
    z_de = -zscore(df["debtToEquity"])  # 负债越低越好

    score = (z_roe + z_roa + z_gm + z_pm + z_de) / 5
    return score.dropna()


def quality(prices: pd.DataFrame, top_n: int = TOP_N,
            freq: str = REBALANCE_FREQ, min_price: float = MIN_PRICE,
            max_w: float = MAX_WEIGHT) -> pd.DataFrame:
    """纯质量因子：ROE / ROA / 毛利率 / 净利率 / 负债率 复合打分
    ⚠️ 用的是当前快照，回测中有 look-ahead bias，仅作快速验证
    """
    q_score = _quality_score()
    valid = prices >= min_price
    scores = {}
    for d in _rebalance_dates(prices, freq):
        s = pd.Series(index=prices.columns, dtype=float)
        common = q_score.index.intersection(prices.columns)
        s[common] = q_score[common]
        s = s.where(valid.loc[d])
        scores[d] = s
    return _apply_weights(prices, scores, top_n, max_w)


def momentum_quality(prices: pd.DataFrame, lookback: int = LOOKBACK_DAYS,
                     top_n: int = TOP_N, freq: str = REBALANCE_FREQ,
                     min_price: float = MIN_PRICE, max_w: float = MAX_WEIGHT,
                     mom_weight: float = 0.5) -> pd.DataFrame:
    """动量 + 质量组合打分（默认各 50%）"""
    q_score = _quality_score()
    mom = prices.pct_change(lookback)
    valid = prices >= min_price

    def zscore(row):
        return (row - row.mean()) / (row.std() + 1e-9)

    scores = {}
    for d in _rebalance_dates(prices, freq):
        v = valid.loc[d]
        # momentum z-score
        z_mom = zscore(mom.loc[d].where(v))
        # quality 静态分数对齐
        q_aligned = pd.Series(index=prices.columns, dtype=float)
        common = q_score.index.intersection(prices.columns)
        q_aligned[common] = q_score[common]
        z_q = zscore(q_aligned.where(v))
        combined = mom_weight * z_mom + (1 - mom_weight) * z_q
        scores[d] = combined
    return _apply_weights(prices, scores, top_n, max_w)


def momentum_quality_voltarget(prices: pd.DataFrame, **kw) -> pd.DataFrame:
    w = momentum_quality(prices)
    return vol_target(w, prices, target_vol=0.15, lookback=30)


def momentum_voltarget(prices: pd.DataFrame, benchmark: pd.Series = None,
                       **kw) -> pd.DataFrame:
    """生产策略：动量 + vol target + 可选的绝对动量过滤
    benchmark: SPY 收盘价 Series，用于绝对动量过滤；None 则自动从 prices 里取 SPY
    """
    w = momentum(
        prices,
        lookback=kw.get("lookback", LOOKBACK_DAYS),
        top_n=kw.get("top_n", TOP_N),
        freq=kw.get("freq", REBALANCE_FREQ),
        min_price=kw.get("min_price", MIN_PRICE),
        max_w=kw.get("max_w", MAX_WEIGHT),
    )
    w = vol_target(
        w, prices,
        target_vol=kw.get("target_vol", TARGET_VOL),
        lookback=kw.get("vol_lookback", VOL_LOOKBACK),
    )
    # 绝对动量 regime 过滤
    if kw.get("abs_mom", ABS_MOM_ENABLED):
        from risk import absolute_momentum_filter
        if benchmark is None and "SPY" in prices.columns:
            benchmark = prices["SPY"]
        if benchmark is not None:
            w = absolute_momentum_filter(
                w, benchmark,
                lookback=kw.get("abs_mom_lookback", ABS_MOM_LOOKBACK),
            )
    return w


def sector_neutral_mom_voltarget(prices: pd.DataFrame, **kw) -> pd.DataFrame:
    w = sector_neutral_momentum(prices)
    return vol_target(w, prices, target_vol=0.15, lookback=30)


STRATEGIES = {
    "momentum": momentum,
    "momentum_12_1": momentum_12_1,
    "momentum_voltarget": momentum_voltarget,
    "quality": quality,
    "momentum_quality": momentum_quality,
    "momentum_quality_voltarget": momentum_quality_voltarget,
    "sector_neutral_mom": sector_neutral_momentum,
    "sector_neutral_mom_vt": sector_neutral_mom_voltarget,
    "multi_factor": multi_factor,
    "risk_parity": risk_parity,
    "mean_reversion": mean_reversion,
    "low_vol": low_vol,
    "dual_ma": dual_ma,
}


def latest_target_weights(prices: pd.DataFrame, strategy: str = "momentum") -> dict[str, float]:
    w = STRATEGIES[strategy](prices)
    last = w.iloc[-1]
    return {k: float(v) for k, v in last[last > 0].items()}
