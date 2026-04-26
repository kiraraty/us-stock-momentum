"""批量下载 S&P 500 + 额外标的的 OHLCV，增量合并到缓存"""
import pandas as pd
from data import (
    get_sp500_tickers, download_ohlcv, _download_batch,
    OHLCV_CACHE, get_field_wide,
)
from config import EXTRA_TICKERS, BENCHMARK, START_DATE, END_DATE


def main():
    sp500 = get_sp500_tickers()
    all_tks = sorted(set(sp500 + EXTRA_TICKERS + [BENCHMARK]))
    print(f"目标总数: {len(all_tks)}  (S&P500={len(sp500)}, 额外={len(EXTRA_TICKERS)})")

    # 已有缓存
    existing = set()
    old_df = None
    if OHLCV_CACHE.exists():
        old_df = pd.read_parquet(OHLCV_CACHE)
        existing = set(old_df.index.get_level_values("ticker").unique())
        print(f"缓存已有 {len(existing)} 只")

    todo = sorted(set(all_tks) - existing)
    if not todo:
        print("无需下载，缓存已齐全")
        return

    print(f"本次下载 {len(todo)} 只: {todo[:10]}{'...' if len(todo)>10 else ''}")
    new_long = download_ohlcv(todo, force=True)  # force 绕过缓存直接拉

    if old_df is not None:
        merged = pd.concat([old_df, new_long]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
    else:
        merged = new_long

    merged.to_parquet(OHLCV_CACHE)
    n = merged.index.get_level_values("ticker").nunique()
    print(f"\n合并完成：{n} 只股票，共 {len(merged):,} 行")

    # 校验缺失
    still_missing = sorted(set(all_tks) - set(merged.index.get_level_values("ticker").unique()))
    if still_missing:
        print(f"⚠️ 仍缺失 {len(still_missing)} 只: {still_missing}")


if __name__ == "__main__":
    main()
