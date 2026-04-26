"""实盘入口：拉数据 → 算目标权重 → 通过 Alpaca 纸面账户调仓
用法：
    python live.py           # 真正下单
    python live.py --dry     # 只打印，不下单
建议配合 cron，每个交易日美东 15:45 运行一次：
    45 15 * * 1-5  cd /path/to/us-stock && /usr/bin/python live.py >> live.log 2>&1
"""
import sys
from data import get_sp500_tickers, download_prices
import pandas as pd
from strategy import momentum_voltarget
from broker import AlpacaBroker
from config import BENCHMARK


def latest_weights(prices: pd.DataFrame, bench: pd.Series) -> dict[str, float]:
    """生产策略：momentum_voltarget（含 vol target + abs mom 过滤）"""
    w = momentum_voltarget(prices, benchmark=bench)
    last = w.iloc[-1]
    return {k: float(v) for k, v in last[last > 0].items()}


def main(dry_run: bool = False):
    tickers = get_sp500_tickers()
    prices = download_prices(tickers + [BENCHMARK], force=False)
    bench = prices[BENCHMARK]
    universe = prices.drop(columns=[BENCHMARK], errors="ignore")

    target = latest_weights(universe, bench)
    print(f"[live] 目标持仓 {len(target)} 只：")
    for s, w in sorted(target.items(), key=lambda x: -x[1]):
        print(f"  {s:<6} {w:.2%}")

    last_prices = universe.iloc[-1].dropna().to_dict()
    broker = AlpacaBroker()
    broker.rebalance_to_weights(target, last_prices, dry_run=dry_run)
    print("[live] 完成")


if __name__ == "__main__":
    main(dry_run="--dry" in sys.argv)
