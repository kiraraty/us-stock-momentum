"""极端事件压力测试
用 momentum_voltarget 在历史极端事件上跑，看策略扛得住什么扛不住什么
⚠️ 存在幸存者偏差，结果偏乐观
"""
import pandas as pd
from data import get_sp500_tickers, download_prices, OHLCV_CACHE, get_field_wide
from strategy import momentum_voltarget
from backtest import run_backtest, compute_metrics
from config import BENCHMARK

# 极端事件定义
EVENTS = [
    ("2000 互联网泡沫",         "2000-01-01", "2002-12-31"),
    ("2003 恢复期",             "2003-01-01", "2006-12-31"),
    ("2007-09 金融危机",        "2007-10-01", "2009-03-31"),
    ("2009 动量崩盘",           "2009-03-01", "2009-12-31"),
    ("2010-11 欧债危机",        "2010-04-01", "2011-12-31"),
    ("2015-16 中国增速放缓",    "2015-01-01", "2016-06-30"),
    ("2018 末崩盘",             "2018-10-01", "2018-12-31"),
    ("2020 疫情闪崩",           "2020-02-01", "2020-04-30"),
    ("2022 通胀加息熊",         "2022-01-01", "2022-12-31"),
]


def annualize(total_return: float, days: int) -> float:
    years = days / 252
    if years < 0.01:
        return total_return
    return (1 + total_return) ** (1 / years) - 1


def main():
    long = pd.read_parquet(OHLCV_CACHE)
    close = get_field_wide(long, "close")
    print(f"数据: {close.shape[0]} 交易日, {close.shape[1]} 股票")
    print(f"日期: {close.index.min().date()} ~ {close.index.max().date()}")

    if BENCHMARK not in close.columns:
        print(f"⚠️ {BENCHMARK} 不在数据里")
        return
    bench = close[BENCHMARK]
    universe = close.drop(columns=[BENCHMARK], errors="ignore")

    # 全程跑一次策略（让动量 lookback 有足够数据）
    print("\n计算全程权重...")
    w = momentum_voltarget(universe)

    print("\n" + "=" * 90)
    print("极端事件压力测试 (momentum_voltarget)")
    print("=" * 90)
    print(f"{'事件':<28} {'策略 CAGR':>10} {'Sharpe':>8} {'MaxDD':>10} {'SPY CAGR':>10} {'超额':>10}")
    print("-" * 90)

    rows = []
    for name, start, end in EVENTS:
        p_seg = universe.loc[start:end]
        if p_seg.empty or p_seg.shape[0] < 10:
            continue
        w_seg = w.loc[start:end]
        b_seg = bench.loc[start:end]
        r = run_backtest(p_seg, w_seg, benchmark=b_seg)
        m = r["metrics"]
        bm = r["benchmark_metrics"]
        alpha = m["CAGR"] - bm["CAGR"]
        rows.append({"event": name, **m, "spy_cagr": bm["CAGR"], "alpha": alpha})
        print(f"{name:<28} {m['CAGR']:>9.2%}  {m['sharpe']:>7.2f}  {m['max_drawdown']:>9.2%}  {bm['CAGR']:>9.2%}  {alpha:>+9.2%}")

    print("\n" + "=" * 90)
    print("全程 (2000-01-01 ~ 2026-04-07)")
    print("=" * 90)
    full_res = run_backtest(universe, w, benchmark=bench)
    fm = full_res["metrics"]
    bm = full_res["benchmark_metrics"]
    print(f"  策略:   CAGR={fm['CAGR']:>7.2%}  Sharpe={fm['sharpe']:>5.2f}  MaxDD={fm['max_drawdown']:>8.2%}  Vol={fm['vol']:>6.2%}")
    print(f"  SPY:    CAGR={bm['CAGR']:>7.2%}  Sharpe={bm['sharpe']:>5.2f}  MaxDD={bm['max_drawdown']:>8.2%}  Vol={bm['vol']:>6.2%}")
    print(f"  超额:   {fm['CAGR']-bm['CAGR']:+.2%}")

    # 每年的收益
    print("\n" + "=" * 90)
    print("按年分解")
    print("=" * 90)
    rets = full_res["returns"]
    yearly = (1 + rets).resample("YE").prod() - 1
    bench_y = (1 + bench.pct_change().fillna(0)).resample("YE").prod() - 1
    bench_y = bench_y.reindex(yearly.index)
    yr = pd.DataFrame({
        "策略":  [f"{x:>7.2%}" for x in yearly.values],
        "SPY":   [f"{x:>7.2%}" for x in bench_y.values],
        "超额":  [f"{(a-b):>+7.2%}" for a,b in zip(yearly.values, bench_y.values)],
    }, index=[d.strftime("%Y") for d in yearly.index])
    print(yr.to_string())

    pd.DataFrame(rows).to_csv("stress_test.csv", index=False)


if __name__ == "__main__":
    main()
