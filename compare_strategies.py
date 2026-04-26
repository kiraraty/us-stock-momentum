"""ETF 轮动 vs S&P 500 个股动量 对比
- 各自独立回测（26 年）
- 相关性分析
- 等权组合回测
- 分段压力测试
"""
import numpy as np
import pandas as pd
from data import OHLCV_CACHE, get_field_wide
from strategy import momentum_voltarget
from etf_strategy import etf_rotation, etf_rotation_voltarget
from backtest import run_backtest, compute_metrics
from config import BENCHMARK


def main():
    long = pd.read_parquet(OHLCV_CACHE)
    close = get_field_wide(long, "close")
    bench = close[BENCHMARK]
    universe = close.drop(columns=[BENCHMARK], errors="ignore")

    print(f"数据: {close.shape[0]} 日, {close.shape[1]} 股票")
    print(f"时期: {close.index.min().date()} ~ {close.index.max().date()}")

    # 生成两个策略的权重
    print("\n[1/3] 计算 S&P 500 个股动量权重...")
    w_stock = momentum_voltarget(universe)

    print("[2/3] 计算 ETF 轮动权重...")
    w_etf = etf_rotation(universe)           # 裸版
    w_etf_vt = etf_rotation_voltarget(universe)  # vol target 版

    # 回测
    print("[3/3] 回测...")
    res_stock = run_backtest(universe, w_stock, benchmark=bench)
    res_etf = run_backtest(universe, w_etf, benchmark=bench)
    res_etf_vt = run_backtest(universe, w_etf_vt, benchmark=bench)

    print("\n" + "=" * 90)
    print("26 年全程对比（2000-2026）")
    print("=" * 90)
    rows = []
    for name, res in [
        ("S&P500 momentum_vt (当前主策略)", res_stock),
        ("ETF rotation (裸跑)", res_etf),
        ("ETF rotation + vol target", res_etf_vt),
        ("SPY 基准", {"metrics": res_stock["benchmark_metrics"]}),
    ]:
        m = res["metrics"]
        rows.append({"策略": name, **m})

    df = pd.DataFrame(rows).set_index("策略")
    disp = df[["total_return", "CAGR", "vol", "sharpe", "max_drawdown", "win_rate"]].copy()
    for c in ["total_return", "CAGR", "vol", "max_drawdown", "win_rate"]:
        disp[c] = disp[c].map(lambda x: f"{x:>9.2%}")
    disp["sharpe"] = disp["sharpe"].map(lambda x: f"{x:>6.2f}")
    pd.set_option("display.width", 200)
    print(disp.to_string())

    # 相关性
    print("\n" + "=" * 90)
    print("策略日收益相关性矩阵")
    print("=" * 90)
    rets_df = pd.DataFrame({
        "stock_mom": res_stock["returns"],
        "etf_rot": res_etf["returns"],
        "etf_rot_vt": res_etf_vt["returns"],
        "SPY": bench.pct_change().fillna(0),
    }).dropna()
    corr = rets_df.corr()
    print(corr.map(lambda x: f"{x:>7.2f}").to_string())

    # 等权组合
    print("\n" + "=" * 90)
    print("等权组合 (50% 个股动量 + 50% ETF 轮动)")
    print("=" * 90)
    combo_ret = 0.5 * res_stock["returns"] + 0.5 * res_etf_vt["returns"]
    combo_eq = (1 + combo_ret).cumprod()
    combo_m = compute_metrics(combo_ret, combo_eq)
    print(f"  CAGR     : {combo_m['CAGR']:>9.2%}")
    print(f"  Sharpe   : {combo_m['sharpe']:>9.2f}")
    print(f"  MaxDD    : {combo_m['max_drawdown']:>9.2%}")
    print(f"  Vol      : {combo_m['vol']:>9.2%}")
    print(f"  胜率     : {combo_m['win_rate']:>9.2%}")

    # 分段压力测试
    print("\n" + "=" * 90)
    print("分段压力测试（momentum_vt / etf_rot_vt / 组合 / SPY）")
    print("=" * 90)
    segs = [
        ("2000-02 互联网泡沫",  "2000-01-01", "2002-12-31"),
        ("2003-06 恢复",         "2003-01-01", "2006-12-31"),
        ("2007-09 金融危机",     "2007-10-01", "2009-03-31"),
        ("2009-03 动量崩盘",     "2009-03-01", "2009-12-31"),
        ("2010-13 震荡复苏",     "2010-01-01", "2013-12-31"),
        ("2014-18 慢牛",         "2014-01-01", "2018-12-31"),
        ("2019-21 疫情放水牛",   "2019-01-01", "2021-12-31"),
        ("2022 通胀熊",          "2022-01-01", "2022-12-31"),
        ("2023-26 AI 牛",        "2023-01-01", "2026-04-07"),
    ]
    print(f"{'时期':<22} {'个股动量':<16} {'ETF轮动':<16} {'组合50/50':<16} {'SPY':<16}")
    print("-" * 90)
    for name, s, e in segs:
        p_seg = universe.loc[s:e]
        if p_seg.empty or p_seg.shape[0] < 20:
            continue
        b_seg = bench.loc[s:e]

        def seg_metrics(rets_series):
            r = rets_series.loc[s:e]
            if len(r) < 2:
                return None
            eq = (1 + r).cumprod()
            m = compute_metrics(r, eq)
            return f"{m['CAGR']:+6.1%}/DD{m['max_drawdown']:+5.1%}"

        stock_s = seg_metrics(res_stock["returns"])
        etf_s = seg_metrics(res_etf_vt["returns"])
        combo_s = seg_metrics(combo_ret)
        bench_r = b_seg.pct_change().fillna(0)
        bench_eq = (1 + bench_r).cumprod()
        bm = compute_metrics(bench_r, bench_eq)
        spy_s = f"{bm['CAGR']:+6.1%}/DD{bm['max_drawdown']:+5.1%}"

        print(f"{name:<22} {stock_s:<16} {etf_s:<16} {combo_s:<16} {spy_s:<16}")


if __name__ == "__main__":
    main()
