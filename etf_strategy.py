"""ETF 行业轮动策略（在本框架实现，便于和个股动量直接对比）
逻辑：
- 池子：20 只美股行业/主题 ETF
- 信号：20 日动量（短周期，和个股动量不同）
- 选 Top N（默认 5）
- 每 5 个交易日调仓
- Point-in-time 筛选：每个调仓日只从"当时已上市"的 ETF 里选
"""
import pandas as pd
from data import OHLCV_CACHE, get_field_wide
from strategy import vol_target


ETF_UNIVERSE = [
    # 行业 SPDR (9 大板块, 2000 年就有)
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU",
    # 指数
    "QQQ",
    # 专项 (不同时期上市)
    "SOXX",    # 2001 半导体
    "IBB",     # 2001 生物科技
    "IYT",     # 2004 运输
    "GLD",     # 2004 黄金
    "GDX",     # 2006 黄金矿业
    "KRE",     # 2006 区域银行
    "XOP",     # 2006 油气勘探
    "XRT",     # 2006 零售
    "XLRE",    # 2015 房地产
    "XLC",     # 2018 通信服务
]


def etf_rotation(prices: pd.DataFrame,
                 lookback: int = 20,
                 top_n: int = 5,
                 rebalance_days: int = 5,
                 min_ret_threshold: float = 0.0) -> pd.DataFrame:
    """ETF 行业轮动
    prices: wide close, 包含所有 ETF 和其他股票也行（只用 ETF 列）
    min_ret_threshold: 动量必须 > 这个阈值才入选（绝对动量过滤，默认 0 = 必须正动量）
    """
    # 只取存在的 ETF
    etfs = [e for e in ETF_UNIVERSE if e in prices.columns]
    px = prices[etfs].copy()

    # N 日动量
    mom = px.pct_change(lookback)

    # 调仓日：每 rebalance_days 个交易日
    rebal_idx = range(lookback + 1, len(px), rebalance_days)
    rebal_dates = [px.index[i] for i in rebal_idx if i < len(px)]

    rebal_rows = {}
    for d in rebal_dates:
        score = mom.loc[d].dropna()
        # 只保留绝对动量 > 阈值的
        score = score[score > min_ret_threshold]
        if score.empty:
            # 全部负动量 → 空仓（全现金）
            rebal_rows[d] = pd.Series(0.0, index=prices.columns)
            continue

        top = score.nlargest(top_n).index
        if len(top) == 0:
            rebal_rows[d] = pd.Series(0.0, index=prices.columns)
            continue

        w = 1.0 / len(top)
        row = pd.Series(0.0, index=prices.columns)
        row[top] = w
        rebal_rows[d] = row

    if not rebal_rows:
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    w_df = pd.DataFrame(rebal_rows).T.sort_index()
    return w_df.reindex(prices.index).ffill().fillna(0.0)


def etf_rotation_voltarget(prices: pd.DataFrame, **kw) -> pd.DataFrame:
    """ETF 轮动 + vol target 包装"""
    w = etf_rotation(prices, **kw)
    return vol_target(w, prices, target_vol=0.18, lookback=30)


if __name__ == "__main__":
    long = pd.read_parquet(OHLCV_CACHE)
    close = get_field_wide(long, "close")
    print(f"数据 shape: {close.shape}")
    print(f"ETF 列在数据里: {[e for e in ETF_UNIVERSE if e in close.columns]}")
    w = etf_rotation(close)
    print(f"权重 shape: {w.shape}")
    print(f"平均持仓数: {(w > 0).sum(axis=1).mean():.1f}")
