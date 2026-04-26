"""一次性跑所有策略并对比"""
import pandas as pd
from data import get_sp500_tickers, download_prices
from strategy import STRATEGIES
from backtest import run_backtest
from config import BENCHMARK


def main():
    tickers = get_sp500_tickers()
    prices = download_prices(tickers + [BENCHMARK])
    bench = prices[BENCHMARK] if BENCHMARK in prices.columns else None
    universe = prices.drop(columns=[BENCHMARK], errors="ignore")

    rows = []
    equities = {}
    for name, fn in STRATEGIES.items():
        print(f"[run] {name} ...")
        w = fn(universe)
        res = run_backtest(universe, w, benchmark=bench)
        m = res["metrics"]
        rows.append({"strategy": name, **m})
        equities[name] = res["equity"]

    if bench is not None:
        br = bench.pct_change().fillna(0)
        be = (1 + br).cumprod()
        from backtest import compute_metrics
        rows.append({"strategy": f"BENCH_{BENCHMARK}", **compute_metrics(br, be)})
        equities[f"BENCH_{BENCHMARK}"] = be

    df = pd.DataFrame(rows).set_index("strategy")
    df = df.sort_values("sharpe", ascending=False)
    print("\n" + "=" * 70)
    print("策略对比（按 Sharpe 降序）")
    print("=" * 70)
    disp = df[["total_return", "CAGR", "vol", "sharpe", "max_drawdown", "win_rate"]].copy()
    for c in ["total_return", "CAGR", "vol", "max_drawdown", "win_rate"]:
        disp[c] = disp[c].map(lambda x: f"{x:>8.2%}")
    disp["sharpe"] = disp["sharpe"].map(lambda x: f"{x:>6.2f}")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(disp.to_string())

    # 保存
    df.to_csv("strategies_compare.csv")
    eq_df = pd.DataFrame(equities)
    eq_df.to_csv("all_equity_curves.csv")
    print("\n结果已保存：strategies_compare.csv, all_equity_curves.csv")


if __name__ == "__main__":
    main()
