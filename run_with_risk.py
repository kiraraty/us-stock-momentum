"""跑带风控版策略，对比裸跑"""
import pandas as pd
from data import get_sp500_tickers, download_prices
from strategy import STRATEGIES
from backtest import run_backtest, compute_metrics
from risk import apply_risk
from config import BENCHMARK


TARGET = ["momentum_voltarget", "momentum", "multi_factor"]


def main():
    tickers = get_sp500_tickers()
    prices = download_prices(tickers + [BENCHMARK])
    bench = prices[BENCHMARK]
    universe = prices.drop(columns=[BENCHMARK], errors="ignore")

    rows = []
    for name in TARGET:
        fn = STRATEGIES[name]
        w_raw = fn(universe)
        w_risk = apply_risk(w_raw, universe, vix_threshold=35.0,
                            dd_threshold=0.15, throttle=0.5)
        for label, w in [(f"{name}", w_raw), (f"{name}+risk", w_risk)]:
            res = run_backtest(universe, w, benchmark=bench)
            m = res["metrics"]
            rows.append({"策略": label, **m})

    # SPY 基准
    br = bench.pct_change().fillna(0)
    be = (1 + br).cumprod()
    rows.append({"策略": "BENCH_SPY", **compute_metrics(br, be)})

    df = pd.DataFrame(rows).set_index("策略")
    disp = df[["total_return", "CAGR", "vol", "sharpe", "max_drawdown", "win_rate"]].copy()
    for c in ["total_return", "CAGR", "vol", "max_drawdown", "win_rate"]:
        disp[c] = disp[c].map(lambda x: f"{x:>8.2%}")
    disp["sharpe"] = disp["sharpe"].map(lambda x: f"{x:>6.2f}")
    pd.set_option("display.width", 200)
    print("=" * 90)
    print("裸跑 vs 加风控（VIX>35切现金 + DD>15%砍半仓）")
    print("=" * 90)
    print(disp.to_string())
    df.to_csv("strategies_with_risk.csv")


if __name__ == "__main__":
    main()
