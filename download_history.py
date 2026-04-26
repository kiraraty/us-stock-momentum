"""扩展历史数据：下载 2000-01-01 ~ 2019-01-01 的 OHLCV，和现有缓存合并
⚠️ 注意：这些老数据会有严重的幸存者偏差——我们只能下载"今天还在 S&P 500 里的"股票
很多 2000-2018 年存在但后来退市/被收购的公司不在池子里
结果应视为"当前幸存者在老时期的表现"而非真正的历史回测
"""
import time
import pandas as pd
import yfinance as yf
from data import (
    get_sp500_tickers, _download_batch, OHLCV_CACHE, DATA_DIR,
)
from config import EXTRA_TICKERS, BENCHMARK

START_HIST = "2000-01-01"
END_HIST = "2019-01-01"


def main():
    sp500 = get_sp500_tickers()
    all_tks = sorted(set(sp500 + EXTRA_TICKERS + [BENCHMARK]))
    print(f"目标 {len(all_tks)} 只")

    # 加载现有缓存
    old_df = None
    if OHLCV_CACHE.exists():
        old_df = pd.read_parquet(OHLCV_CACHE)
        print(f"现有缓存: {len(old_df):,} 行, 日期 {old_df.index.get_level_values('date').min().date()} ~ {old_df.index.get_level_values('date').max().date()}")

    print(f"\n下载 {START_HIST} ~ {END_HIST}")
    batch_size = 20
    n = len(all_tks)
    n_batches = (n - 1) // batch_size + 1

    frames = []
    failed = []
    for i in range(0, n, batch_size):
        batch = all_tks[i:i + batch_size]
        bi = i // batch_size + 1
        print(f"  批次 {bi}/{n_batches}  ({batch[0]}..{batch[-1]})")
        df = _download_batch(batch, START_HIST, END_HIST)
        if not df.empty:
            got = df.index.get_level_values("ticker").unique()
            frames.append(df)
            print(f"    ✓ {len(got)}/{len(batch)}")
        else:
            failed.extend(batch)
        time.sleep(2.5)

    if not frames:
        raise RuntimeError("全部失败")

    new_long = pd.concat(frames).sort_index()
    new_long = new_long[~new_long.index.duplicated()]
    print(f"\n新下载 {len(new_long):,} 行")

    # 合并
    if old_df is not None:
        merged = pd.concat([old_df, new_long]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
    else:
        merged = new_long

    merged.to_parquet(OHLCV_CACHE)
    dates = merged.index.get_level_values("date")
    tks_got = merged.index.get_level_values("ticker").nunique()
    size_mb = OHLCV_CACHE.stat().st_size / 1e6
    print(f"\n合并完成 → {OHLCV_CACHE}  {size_mb:.1f} MB")
    print(f"  股票数: {tks_got}")
    print(f"  日期范围: {dates.min().date()} ~ {dates.max().date()}")
    print(f"  总行数: {len(merged):,}")
    if failed:
        print(f"  失败: {failed}")


if __name__ == "__main__":
    main()
