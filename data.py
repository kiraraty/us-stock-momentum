"""数据层：S&P 500 成分股 + yfinance OHLCV 历史行情
- 存储格式：long format parquet（date, ticker, open, high, low, close, volume）
- 风控：分批下载 + sleep + 指数回退重试 + 二次回收
- 向后兼容：`download_prices()` 仍返回 wide-close
"""
import io
import time
import random
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime

from config import DATA_DIR, START_DATE, END_DATE

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36")

OHLCV_CACHE = DATA_DIR / "ohlcv.parquet"
LEGACY_CLOSE_CACHE = DATA_DIR / "prices.parquet"   # 老的 close-only 缓存


# ========= S&P 500 成分股 =========
def get_sp500_tickers() -> list[str]:
    return get_sp500_info()["ticker"].tolist()


def get_sp500_info() -> pd.DataFrame:
    """返回 S&P 500 成分股 + 行业信息（ticker, sector, industry）"""
    cache = DATA_DIR / "sp500_info.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    for i in range(3):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
            r.raise_for_status()
            df = pd.read_html(io.StringIO(r.text))[0]
            out = pd.DataFrame({
                "ticker": df["Symbol"].str.replace(".", "-", regex=False),
                "sector": df["GICS Sector"],
                "industry": df["GICS Sub-Industry"],
            })
            out.to_parquet(cache)
            return out
        except Exception as e:
            print(f"[data] 取 wiki 失败 ({i+1}/3): {str(e)[:80]}")
            time.sleep(2)

    # 兜底：老缓存只有 ticker，没有 sector
    legacy = DATA_DIR / "sp500_tickers.parquet"
    if legacy.exists():
        print("[data] 回退到老的 ticker-only 缓存，sector 标记为 Unknown")
        df = pd.read_parquet(legacy)
        return pd.DataFrame({
            "ticker": df["ticker"],
            "sector": "Unknown",
            "industry": "Unknown",
        })
    raise RuntimeError("无法获取 S&P 500 成分股")


def get_sector_map() -> dict[str, str]:
    info = get_sp500_info()
    return dict(zip(info["ticker"], info["sector"]))


# ========= 行情下载（带重试） =========
def _download_batch(tickers: list[str], start: str, end: str,
                    retries: int = 4) -> pd.DataFrame:
    """返回 long format: index=(date, ticker), columns=[open,high,low,close,volume]"""
    for i in range(retries):
        try:
            raw = yf.download(
                tickers, start=start, end=end,
                auto_adjust=True, progress=False, threads=True,
                group_by="ticker",
            )
            if raw is None or raw.empty:
                raise RuntimeError("空返回")

            # raw 是 MultiIndex 列: (ticker, field)
            if isinstance(raw.columns, pd.MultiIndex):
                # stack ticker 这一层 → 行索引多一层 ticker
                long = raw.stack(level=0, future_stack=True)
                long.index.names = ["date", "ticker"]
            else:
                # 单票情况
                long = raw.copy()
                long["ticker"] = tickers[0]
                long = long.reset_index().set_index(["Date", "ticker"])
                long.index.names = ["date", "ticker"]

            long.columns = [c.lower() for c in long.columns]
            keep = ["open", "high", "low", "close", "volume"]
            long = long[[c for c in keep if c in long.columns]]
            long = long.dropna(subset=["close"])
            if long.empty:
                raise RuntimeError("全 NaN")
            return long
        except Exception as e:
            wait = (2 ** i) * 3 + random.uniform(0, 2)
            print(f"  ! 第{i+1}次失败: {str(e)[:100]}，{wait:.1f}s 后重试")
            time.sleep(wait)
    print(f"  ✗ 放弃批次（{len(tickers)}只）")
    return pd.DataFrame()


def download_ohlcv(tickers: list[str], start: str = START_DATE,
                   end: str | None = END_DATE, force: bool = False,
                   batch_size: int = 20, batch_sleep: float = 2.5) -> pd.DataFrame:
    """下载 OHLCV，long format"""
    if OHLCV_CACHE.exists() and not force:
        df = pd.read_parquet(OHLCV_CACHE)
        dates = df.index.get_level_values("date")
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) if end else pd.Timestamp.today()
        if dates.min() <= start_ts + pd.Timedelta(days=7) and \
           dates.max() >= end_ts - pd.Timedelta(days=7):
            print(f"[data] 使用缓存 {OHLCV_CACHE} rows={len(df)}")
            return df

    end = end or datetime.today().strftime("%Y-%m-%d")
    tickers = sorted(set(tickers))
    n_batches = (len(tickers) - 1) // batch_size + 1
    print(f"[data] 下载 OHLCV {len(tickers)} 只  {start} ~ {end}  batch={batch_size}  共 {n_batches} 批")

    frames = []
    failed = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        bi = i // batch_size + 1
        print(f"  批次 {bi}/{n_batches}  ({batch[0]}..{batch[-1]})")
        df = _download_batch(batch, start, end)
        if not df.empty:
            got = df.index.get_level_values("ticker").unique()
            frames.append(df)
            missing = set(batch) - set(got)
            if missing:
                print(f"    ✓ {len(got)}/{len(batch)}, 缺: {sorted(missing)[:3]}{'...' if len(missing)>3 else ''}")
            else:
                print(f"    ✓ {len(got)}/{len(batch)}")
        else:
            failed.extend(batch)
        time.sleep(batch_sleep)

    if not frames:
        raise RuntimeError("所有批次均下载失败")

    all_long = pd.concat(frames).sort_index()
    all_long = all_long[~all_long.index.duplicated()]

    # 二次回收
    got = set(all_long.index.get_level_values("ticker").unique())
    missing = sorted(set(tickers) - got)
    if missing:
        print(f"\n[data] 二次回收 {len(missing)} 只（单只串行）")
        recovered = []
        for j, t in enumerate(missing, 1):
            print(f"  [{j}/{len(missing)}] {t}", end=" ")
            df = _download_batch([t], start, end, retries=3)
            if not df.empty:
                recovered.append(df)
                print("✓")
            else:
                print("✗")
                failed.append(t)
            time.sleep(0.8)
        if recovered:
            extra = pd.concat(recovered).sort_index()
            all_long = pd.concat([all_long, extra]).sort_index()
            all_long = all_long[~all_long.index.duplicated()]

    all_long.to_parquet(OHLCV_CACHE)
    n_tickers = all_long.index.get_level_values("ticker").nunique()
    n_dates = all_long.index.get_level_values("date").nunique()
    size_mb = OHLCV_CACHE.stat().st_size / 1e6
    print(f"\n[data] 完成 → {OHLCV_CACHE}  {size_mb:.1f} MB")
    print(f"       {n_tickers} 只股票  {n_dates} 个交易日  总计 {len(all_long):,} 行")
    if failed:
        print(f"       失败: {failed}")
    return all_long


# ========= 视图 helper =========
def get_field_wide(long: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """long → wide: index=date, columns=ticker, 值=field"""
    return long[field].unstack("ticker").sort_index()


def download_prices(tickers: list[str], start: str = START_DATE,
                    end: str | None = END_DATE, force: bool = False) -> pd.DataFrame:
    """向后兼容：返回 wide-close"""
    # 如果老缓存还在且足够新，优先用（避免首次重下）
    if LEGACY_CLOSE_CACHE.exists() and not force and not OHLCV_CACHE.exists():
        df = pd.read_parquet(LEGACY_CLOSE_CACHE)
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) if end else pd.Timestamp.today()
        if df.index.min() <= start_ts + pd.Timedelta(days=7) and \
           df.index.max() >= end_ts - pd.Timedelta(days=7):
            print(f"[data] (legacy) 使用 close-only 缓存 shape={df.shape}")
            return df

    long = download_ohlcv(tickers, start, end, force=force)
    return get_field_wide(long, "close")


if __name__ == "__main__":
    tks = get_sp500_tickers()
    print(f"S&P 500 共 {len(tks)} 只")
    long = download_ohlcv(tks + ["SPY"], force=True)
    print("\n样例（AAPL 最近 5 天）:")
    print(long.xs("AAPL", level="ticker").tail())
    close = get_field_wide(long, "close")
    vol = get_field_wide(long, "volume")
    print(f"\nclose wide shape={close.shape}   volume wide shape={vol.shape}")
