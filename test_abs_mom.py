"""绝对动量过滤 - 26 年压力测试"""
import pandas as pd
from data import OHLCV_CACHE, get_field_wide
from strategy import momentum_voltarget
from risk import absolute_momentum_filter
from backtest import run_backtest, compute_metrics
from config import BENCHMARK


def yearly(r):
    return (1 + r).resample("YE").prod() - 1


def main():
    long = pd.read_parquet(OHLCV_CACHE)
    close = get_field_wide(long, "close")
    bench = close[BENCHMARK]
    uni = close.drop(columns=[BENCHMARK], errors="ignore")

    print(f"数据: {close.shape}  时期: {close.index.min().date()} ~ {close.index.max().date()}\n")

    # 1. 基线
    w_base = momentum_voltarget(uni)
    r_base = run_backtest(uni, w_base, benchmark=bench)

    # 2. 不同 lookback 的绝对动量过滤
    configs = [
        ("基线 (无过滤)",             w_base),
        ("+ AbsMom 6月 (126d)",      absolute_momentum_filter(w_base, bench, lookback=126)),
        ("+ AbsMom 9月 (189d)",      absolute_momentum_filter(w_base, bench, lookback=189)),
        ("+ AbsMom 12月 (252d)",     absolute_momentum_filter(w_base, bench, lookback=252)),
        ("+ AbsMom 10月 (210d)",     absolute_momentum_filter(w_base, bench, lookback=210)),
    ]

    results = {}
    print("=" * 90)
    print("全程 26 年对比")
    print("=" * 90)
    print(f"{'配置':<30} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>10} {'Vol':>8} {'胜率':>8}")
    print("-" * 90)
    for name, w in configs:
        res = run_backtest(uni, w, benchmark=bench)
        m = res["metrics"]
        results[name] = res
        print(f"{name:<30} {m['CAGR']:>7.2%} {m['sharpe']:>8.2f} {m['max_drawdown']:>9.2%} {m['vol']:>7.2%} {m['win_rate']:>7.2%}")

    # SPY 基准
    spy_r = bench.pct_change().fillna(0)
    spy_eq = (1 + spy_r).cumprod()
    sm = compute_metrics(spy_r, spy_eq)
    print(f"{'SPY 基准':<30} {sm['CAGR']:>7.2%} {sm['sharpe']:>8.2f} {sm['max_drawdown']:>9.2%} {sm['vol']:>7.2%} {sm['win_rate']:>7.2%}")

    # 年度对比（选 12 月版本和基线）
    print("\n" + "=" * 90)
    print("年度收益对比（基线 vs AbsMom 12月）")
    print("=" * 90)
    base = yearly(results["基线 (无过滤)"]["returns"])
    am = yearly(results["+ AbsMom 12月 (252d)"]["returns"])
    spy_y = yearly(spy_r).reindex(base.index)
    print(f"{'年份':<8} {'基线':>10} {'+AbsMom':>10} {'SPY':>10} {'AbsMom vs 基线':>16}")
    print("-" * 60)
    notes = {
        "2001": "互联网崩1", "2002": "互联网崩2", "2008": "金融危机",
        "2009": "动量崩盘", "2018": "Fed急转弯", "2020": "疫情",
        "2022": "通胀加息",
    }
    for d, (b, a, s) in zip(base.index, zip(base.values, am.values, spy_y.values)):
        y = d.strftime("%Y")
        diff = a - b
        note = notes.get(y, "")
        marker = " ←" if abs(diff) > 0.05 else ""
        spy_str = f"{s:>10.1%}" if pd.notna(s) else "     -- "
        print(f"{y:<8} {b:>10.1%} {a:>10.1%} {spy_str} {diff:>+15.1%}  {note}{marker}")

    # 分段压力测试
    print("\n" + "=" * 90)
    print("极端事件分段对比")
    print("=" * 90)
    segs = [
        ("2000-02 互联网", "2000-01-01", "2002-12-31"),
        ("2007-09 金融危机", "2007-10-01", "2009-03-31"),
        ("2009 动量崩盘",  "2009-03-01", "2009-12-31"),
        ("2018 末崩盘",    "2018-10-01", "2018-12-31"),
        ("2020 疫情",      "2020-02-01", "2020-04-30"),
        ("2022 通胀熊",    "2022-01-01", "2022-12-31"),
    ]
    print(f"{'时期':<22} {'基线':>20} {'+AbsMom 12月':>20}")
    print("-" * 70)
    def seg_fmt(rets, s, e):
        r = rets.loc[s:e]
        if len(r) < 2:
            return "     --"
        eq = (1 + r).cumprod()
        m = compute_metrics(r, eq)
        return f"{m['CAGR']:+6.1%} / DD{m['max_drawdown']:+5.1%}"

    for name, s, e in segs:
        base_s = seg_fmt(results["基线 (无过滤)"]["returns"], s, e)
        am_s = seg_fmt(results["+ AbsMom 12月 (252d)"]["returns"], s, e)
        print(f"{name:<22} {base_s:>20} {am_s:>20}")


if __name__ == "__main__":
    main()
