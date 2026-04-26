"""Walk-forward 分段验证：把 2019-2026 切三段看策略稳定性
- 段 1: 2019-01 ~ 2021-12  (科技牛市)
- 段 2: 2022-01 ~ 2022-12  (熊市 / 加息)
- 段 3: 2023-01 ~ 2026-04  (震荡回升)
"""
import pandas as pd
from data import get_sp500_tickers, download_prices
from strategy import STRATEGIES
from backtest import run_backtest, compute_metrics
from config import BENCHMARK

SEGMENTS = [
    ("2019 科技牛市",    "2019-01-01", "2021-12-31"),
    ("2022 熊市",        "2022-01-01", "2022-12-31"),
    ("2023+ 震荡回升",    "2023-01-01", "2026-04-08"),
    ("全程",              "2019-01-01", "2026-04-08"),
]

STRATEGIES_TO_TEST = ["momentum_voltarget", "momentum_quality_voltarget", "momentum_quality", "quality"]


def main():
    tickers = get_sp500_tickers()
    prices = download_prices(tickers + [BENCHMARK])
    bench = prices[BENCHMARK]
    universe = prices.drop(columns=[BENCHMARK], errors="ignore")

    rows = []
    for name in STRATEGIES_TO_TEST:
        fn = STRATEGIES[name]
        # 先在全程上算权重（避免边界 look-back 效应），再分段切收益
        w_all = fn(universe)
        for seg_name, start, end in SEGMENTS:
            p_seg = universe.loc[start:end]
            w_seg = w_all.loc[start:end]
            b_seg = bench.loc[start:end]
            if p_seg.empty:
                continue
            res = run_backtest(p_seg, w_seg, benchmark=b_seg)
            m = res["metrics"]
            bm = res["benchmark_metrics"]
            rows.append({
                "strategy": name,
                "segment": seg_name,
                "CAGR": m["CAGR"],
                "sharpe": m["sharpe"],
                "max_dd": m["max_drawdown"],
                "SPY_CAGR": bm["CAGR"],
                "SPY_sharpe": bm["sharpe"],
                "alpha_CAGR": m["CAGR"] - bm["CAGR"],
            })

    df = pd.DataFrame(rows)
    print("=" * 100)
    print("Walk-forward 分段结果")
    print("=" * 100)
    for strat in STRATEGIES_TO_TEST:
        print(f"\n### {strat}")
        sub = df[df["strategy"] == strat].drop(columns=["strategy"]).reset_index(drop=True)
        disp = sub.copy()
        for c in ["CAGR", "max_dd", "SPY_CAGR", "alpha_CAGR"]:
            disp[c] = disp[c].map(lambda x: f"{x:>8.2%}")
        for c in ["sharpe", "SPY_sharpe"]:
            disp[c] = disp[c].map(lambda x: f"{x:>6.2f}")
        print(disp.to_string(index=False))

    df.to_csv("walk_forward.csv", index=False)
    print("\n已保存: walk_forward.csv")


if __name__ == "__main__":
    main()
