"""向量化回测 + 指标"""
import numpy as np
import pandas as pd
from config import SLIPPAGE_BPS, BENCHMARK


def run_backtest(prices: pd.DataFrame, weights: pd.DataFrame,
                 benchmark: pd.Series | None = None) -> dict:
    """
    prices: wide 收盘价
    weights: 每日目标权重（T 日收盘算出，T+1 开盘后生效 → 这里简化为 T+1 收盘生效）
    """
    # 对齐
    tickers = [t for t in weights.columns if t in prices.columns]
    prices = prices[tickers].ffill()
    weights = weights[tickers].reindex(prices.index).fillna(0)

    # 日收益
    rets = prices.pct_change().fillna(0)

    # T 日权重 → T+1 生效
    w_shift = weights.shift(1).fillna(0)

    # 组合日收益
    port_ret = (w_shift * rets).sum(axis=1)

    # 换手成本
    turnover = (weights - w_shift).abs().sum(axis=1)
    cost = turnover * (SLIPPAGE_BPS / 1e4)
    port_ret = port_ret - cost

    equity = (1 + port_ret).cumprod()

    metrics = compute_metrics(port_ret, equity)

    result = {
        "equity": equity,
        "returns": port_ret,
        "metrics": metrics,
        "turnover": turnover,
    }
    if benchmark is not None:
        bench_ret = benchmark.pct_change().reindex(port_ret.index).fillna(0)
        bench_eq = (1 + bench_ret).cumprod()
        result["benchmark_equity"] = bench_eq
        result["benchmark_metrics"] = compute_metrics(bench_ret, bench_eq)
    return result


def compute_metrics(returns: pd.Series, equity: pd.Series) -> dict:
    n = len(returns)
    if n == 0:
        return {}
    total_ret = equity.iloc[-1] - 1
    years = n / 252
    cagr = (equity.iloc[-1]) ** (1 / years) - 1 if years > 0 else 0
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / vol if vol > 0 else 0
    dd = equity / equity.cummax() - 1
    max_dd = dd.min()
    win = (returns > 0).mean()
    return {
        "total_return": float(total_ret),
        "CAGR": float(cagr),
        "vol": float(vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": float(win),
    }


def print_metrics(name: str, m: dict):
    print(f"\n===== {name} =====")
    print(f"  总收益:    {m['total_return']:>8.2%}")
    print(f"  年化(CAGR):{m['CAGR']:>8.2%}")
    print(f"  年化波动:  {m['vol']:>8.2%}")
    print(f"  夏普:      {m['sharpe']:>8.2f}")
    print(f"  最大回撤:  {m['max_drawdown']:>8.2%}")
    print(f"  胜率:      {m['win_rate']:>8.2%}")


if __name__ == "__main__":
    from data import get_sp500_tickers, download_prices
    from strategy import momentum_weights

    tickers = get_sp500_tickers()
    prices = download_prices(tickers + [BENCHMARK])
    bench = prices[BENCHMARK] if BENCHMARK in prices.columns else None
    universe = prices.drop(columns=[BENCHMARK], errors="ignore")

    w = momentum_weights(universe)
    res = run_backtest(universe, w, benchmark=bench)

    print_metrics("动量策略", res["metrics"])
    if "benchmark_metrics" in res:
        print_metrics(f"基准 {BENCHMARK}", res["benchmark_metrics"])

    # 保存曲线
    out = pd.DataFrame({"strategy": res["equity"]})
    if "benchmark_equity" in res:
        out["benchmark"] = res["benchmark_equity"]
    out.to_csv("equity_curve.csv")
    print("\n净值曲线已保存到 equity_curve.csv")
