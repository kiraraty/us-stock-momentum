"""基本面快照下载（方案 A）
- 用 yfinance Ticker.info 取 ROE、D/E、毛利率等
- 单只串行 + sleep + 重试 + 增量缓存（崩了能续）
- ⚠️ 这是当前快照，回测中会有 look-ahead bias，仅作快速验证用
"""
import time
import random
import pandas as pd
import yfinance as yf

from data import get_sp500_tickers, DATA_DIR
from config import EXTRA_TICKERS, BENCHMARK

CACHE = DATA_DIR / "fundamentals.parquet"

FIELDS = [
    "returnOnEquity",      # ROE
    "returnOnAssets",      # ROA
    "debtToEquity",        # 负债率
    "profitMargins",       # 净利率
    "grossMargins",        # 毛利率
    "operatingMargins",    # 营业利润率
    "trailingPE",          # PE
    "forwardPE",
    "priceToBook",         # PB
    "earningsGrowth",      # 盈利增速
    "revenueGrowth",       # 营收增速
    "currentRatio",        # 流动比率
    "marketCap",
]


def fetch_one(ticker: str, retries: int = 3) -> dict | None:
    for i in range(retries):
        try:
            t = yf.Ticker(ticker)
            info = t.info
            if not info or len(info) < 5:
                raise RuntimeError("空 info")
            row = {"ticker": ticker}
            for f in FIELDS:
                row[f] = info.get(f)
            return row
        except Exception as e:
            wait = (2 ** i) + random.uniform(0, 1)
            if i == retries - 1:
                return None
            time.sleep(wait)
    return None


def download_fundamentals(tickers: list[str], force: bool = False, sleep: float = 0.4):
    """串行下载，每只 sleep 一下，过程持久化"""
    existing = pd.DataFrame(columns=["ticker"] + FIELDS)
    if CACHE.exists() and not force:
        existing = pd.read_parquet(CACHE)
        done = set(existing["ticker"])
        tickers = [t for t in tickers if t not in done]
        print(f"[fund] 缓存已有 {len(done)}，本次下载 {len(tickers)}")

    if not tickers:
        print("[fund] 无需下载")
        return existing

    rows = existing.to_dict("records")
    n = len(tickers)
    save_every = 25
    fail = []

    for i, t in enumerate(tickers, 1):
        row = fetch_one(t)
        if row is None:
            fail.append(t)
            print(f"  [{i}/{n}] {t} ✗")
        else:
            rows.append(row)
            roe = row.get("returnOnEquity")
            roe_str = f"{roe:.2%}" if isinstance(roe, (int, float)) else "—"
            print(f"  [{i}/{n}] {t} ✓ ROE={roe_str}")

        # 定期保存
        if i % save_every == 0:
            pd.DataFrame(rows).to_parquet(CACHE)
            print(f"  --- 已保存 {len(rows)} 条 ---")
        time.sleep(sleep)

    df = pd.DataFrame(rows)
    df.to_parquet(CACHE)
    print(f"\n[fund] 完成，共 {len(df)} 条，失败 {len(fail)}")
    if fail:
        print(f"  失败: {fail}")
    return df


def load_fundamentals() -> pd.DataFrame:
    return pd.read_parquet(CACHE).set_index("ticker")


if __name__ == "__main__":
    tks = get_sp500_tickers()
    all_tks = sorted(set(tks + EXTRA_TICKERS + [BENCHMARK]))
    print(f"目标 {len(all_tks)} 只")
    df = download_fundamentals(all_tks)
    print("\n样例:")
    print(df[["ticker", "returnOnEquity", "debtToEquity", "profitMargins"]].head(10))
