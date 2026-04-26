"""Optuna 极限版：放开所有约束，看回测能挤出多少
⚠️ 这是 in-sample optimization，结果几乎必然过拟合，仅作上限参考
"""
import optuna
import pandas as pd
from data import get_sp500_tickers, download_prices
from strategy import momentum, vol_target
from backtest import run_backtest
from config import BENCHMARK

optuna.logging.set_verbosity(optuna.logging.WARNING)


def main(n_trials: int = 200):
    tickers = get_sp500_tickers()
    prices = download_prices(tickers + [BENCHMARK])
    bench = prices[BENCHMARK]
    universe = prices.drop(columns=[BENCHMARK], errors="ignore")

    def objective(trial: optuna.Trial):
        lookback = trial.suggest_int("lookback", 20, 400)
        top_n = trial.suggest_int("top_n", 2, 50)               # 下探到 2
        target_vol = trial.suggest_float("target_vol", 0.05, 1.0)  # 放开到 100%
        vol_lookback = trial.suggest_int("vol_lookback", 5, 120)
        freq = trial.suggest_categorical("freq", ["W-FRI", "2W-FRI", "M", "BMS", "Q"])
        max_w = trial.suggest_float("max_w", 0.05, 1.0)          # 单票允许 100%

        try:
            w_raw = momentum(universe, lookback=lookback, top_n=top_n,
                             freq=freq, max_w=max_w)
            w = vol_target(w_raw, universe, target_vol=target_vol, lookback=vol_lookback)
            res = run_backtest(universe, w, benchmark=bench)
            return res["metrics"]["CAGR"]   # 极限 CAGR 版
        except Exception:
            return -10.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n=== 完成 {n_trials} 次试验 ===")
    print(f"最佳 Sharpe: {study.best_value:.3f}")
    print(f"最佳参数: {study.best_params}")

    p = study.best_params
    w = vol_target(
        momentum(universe, lookback=p["lookback"], top_n=p["top_n"],
                 freq=p["freq"], max_w=p["max_w"]),
        universe, target_vol=p["target_vol"], lookback=p["vol_lookback"]
    )
    res = run_backtest(universe, w, benchmark=bench)
    m = res["metrics"]
    print(f"\n--- 全程表现 ---")
    print(f"  CAGR     : {m['CAGR']:.2%}")
    print(f"  Sharpe   : {m['sharpe']:.2f}")
    print(f"  Max DD   : {m['max_drawdown']:.2%}")
    print(f"  Vol      : {m['vol']:.2%}")
    print(f"  胜率     : {m['win_rate']:.2%}")
    print(f"  累计收益 : {m['total_return']:.2%}")

    # 分段看是不是真的能打
    print("\n--- 分段（看是否过拟合）---")
    segs = [
        ("2019-21", "2019-01-01", "2021-12-31"),
        ("2022   ", "2022-01-01", "2022-12-31"),
        ("2023+  ", "2023-01-01", "2026-04-08"),
    ]
    for name, s, e in segs:
        ps = universe.loc[s:e]; ws = w.loc[s:e]; bs = bench.loc[s:e]
        r = run_backtest(ps, ws, benchmark=bs)
        sm = r["metrics"]; bm = r["benchmark_metrics"]
        a = sm["CAGR"] - bm["CAGR"]
        print(f"  {name}: CAGR={sm['CAGR']:>7.2%}  Sharpe={sm['sharpe']:>5.2f}  DD={sm['max_drawdown']:>7.2%}  alpha={a:>+6.2%}")

    print("\n--- Top 15 试验 ---")
    df = study.trials_dataframe()
    cols = ["value"] + [c for c in df.columns if c.startswith("params_")]
    print(df.nlargest(15, "value")[cols].to_string(index=False))


if __name__ == "__main__":
    main(n_trials=200)
