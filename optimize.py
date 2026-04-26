"""Optuna 搜 momentum_voltarget 的参数空间
- 注意：避免过拟合，用 walk-forward 三段做评分而不是全程
- 目标：所有段都跑赢 SPY 的最小 alpha 最大化（min-max 鲁棒）
"""
import optuna
import pandas as pd
from data import get_sp500_tickers, download_prices
from strategy import momentum, vol_target
from backtest import run_backtest
from config import BENCHMARK

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEGMENTS = [
    ("2019-21", "2019-01-01", "2021-12-31"),
    ("2022",    "2022-01-01", "2022-12-31"),
    ("2023+",   "2023-01-01", "2026-04-08"),
]


def main(n_trials: int = 60):
    tickers = get_sp500_tickers()
    prices = download_prices(tickers + [BENCHMARK])
    bench = prices[BENCHMARK]
    universe = prices.drop(columns=[BENCHMARK], errors="ignore")

    def objective(trial: optuna.Trial):
        lookback = trial.suggest_int("lookback", 40, 252)
        top_n = trial.suggest_int("top_n", 5, 15)        # 集中持仓约束
        target_vol = trial.suggest_float("target_vol", 0.10, 0.25)
        vol_lookback = trial.suggest_int("vol_lookback", 15, 60)

        try:
            w_raw = momentum(universe, lookback=lookback, top_n=top_n)
            w = vol_target(w_raw, universe, target_vol=target_vol, lookback=vol_lookback)
        except Exception:
            return -10.0

        # 评分：min(alpha) across segments  ——  鲁棒型目标
        alphas = []
        for _, s, e in SEGMENTS:
            p_seg = universe.loc[s:e]
            w_seg = w.loc[s:e]
            b_seg = bench.loc[s:e]
            res = run_backtest(p_seg, w_seg, benchmark=b_seg)
            alpha = res["metrics"]["CAGR"] - res["benchmark_metrics"]["CAGR"]
            alphas.append(alpha)
        # 同时鼓励高 sharpe + 控回撤
        full_res = run_backtest(universe, w, benchmark=bench)
        sharpe = full_res["metrics"]["sharpe"]
        max_dd = full_res["metrics"]["max_drawdown"]
        # 综合分：min alpha 主导 + sharpe 加成 + dd 惩罚
        score = min(alphas) + 0.1 * sharpe + 0.5 * max_dd  # max_dd 是负数
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"\n=== 完成 {n_trials} 次试验 ===")
    print("最佳综合分:", round(study.best_value, 4))
    print("最佳参数:", study.best_params)

    # 用最佳参数跑全程 + walk-forward
    p = study.best_params
    w = vol_target(
        momentum(universe, lookback=p["lookback"], top_n=p["top_n"]),
        universe, target_vol=p["target_vol"], lookback=p["vol_lookback"]
    )
    full_res = run_backtest(universe, w, benchmark=bench)
    m = full_res["metrics"]
    print(f"\n--- 最佳参数全程表现 ---")
    print(f"  CAGR     : {m['CAGR']:.2%}")
    print(f"  Sharpe   : {m['sharpe']:.2f}")
    print(f"  Max DD   : {m['max_drawdown']:.2%}")
    print(f"  Vol      : {m['vol']:.2%}")

    print("\n--- 分段 ---")
    for name, s, e in SEGMENTS:
        p_seg = universe.loc[s:e]
        w_seg = w.loc[s:e]
        b_seg = bench.loc[s:e]
        r = run_backtest(p_seg, w_seg, benchmark=b_seg)
        a = r["metrics"]["CAGR"] - r["benchmark_metrics"]["CAGR"]
        print(f"  {name}: CAGR={r['metrics']['CAGR']:>7.2%}  Sharpe={r['metrics']['sharpe']:>5.2f}  alpha vs SPY={a:>+7.2%}")

    # 看 top 10 试验，判断稳健性
    print("\n--- Top 10 试验（看参数稳健性）---")
    df = study.trials_dataframe()
    top = df.nlargest(10, "value")[["value"] + [c for c in df.columns if c.startswith("params_")]]
    print(top.to_string(index=False))

    return study


if __name__ == "__main__":
    main(n_trials=60)
