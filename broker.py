"""Alpaca 纸面交易封装（alpaca-py SDK）"""
import time
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER


class AlpacaBroker:
    def __init__(self):
        if not ALPACA_API_KEY:
            raise RuntimeError("请在 .env 中配置 ALPACA_API_KEY / ALPACA_SECRET_KEY")
        self.client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)

    def account(self):
        a = self.client.get_account()
        return {
            "equity": float(a.equity),
            "cash": float(a.cash),
            "buying_power": float(a.buying_power),
            "status": a.status,
        }

    def positions(self) -> dict[str, float]:
        """返回 {symbol: market_value}"""
        return {p.symbol: float(p.market_value) for p in self.client.get_all_positions()}

    def position_qty(self) -> dict[str, float]:
        return {p.symbol: float(p.qty) for p in self.client.get_all_positions()}

    def submit_market(self, symbol: str, qty: float, side: OrderSide):
        if qty <= 0:
            return None
        req = MarketOrderRequest(
            symbol=symbol,
            qty=round(qty, 4),           # Alpaca 支持零股（fractional）
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        # Alpaca API 偶尔超时，重试 3 次
        for attempt in range(3):
            try:
                return self.client.submit_order(req)
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    raise

    def rebalance_to_weights(self, target_weights: dict[str, float],
                             last_prices: dict[str, float], dry_run: bool = False):
        """按目标权重下差额市价单"""
        acc = self.account()
        equity = acc["equity"]
        print(f"[broker] 账户净值={equity:.2f} 现金={acc['cash']:.2f}")

        cur_val = self.positions()
        target_val = {s: equity * w for s, w in target_weights.items()}

        all_syms = set(cur_val) | set(target_val)
        orders = []
        for sym in sorted(all_syms):
            tv = target_val.get(sym, 0.0)
            cv = cur_val.get(sym, 0.0)
            diff_val = tv - cv
            px = last_prices.get(sym)
            if px is None or px <= 0:
                continue
            qty = abs(diff_val) / px
            if qty * px < 1.0:  # 差额 < $1 忽略
                continue
            side = OrderSide.BUY if diff_val > 0 else OrderSide.SELL
            orders.append((sym, qty, side, diff_val))

        for sym, qty, side, diff in orders:
            print(f"  {side.value:>4} {sym:<6} qty={qty:>10.4f}  (~${diff:+.0f})")
            if not dry_run:
                try:
                    self.submit_market(sym, qty, side)
                except Exception as e:
                    print(f"    ✗ 失败: {e}")
        return orders


if __name__ == "__main__":
    b = AlpacaBroker()
    print(b.account())
    print("当前持仓:", b.positions())
