import numpy as np
import pandas as pd
from datetime import datetime, time
from typing import Optional

name = "Trigger Happy Elite v0.5"
timeframe = "5m"
leverage = 1

class TriggerHappyElite:
    """Stateful strategy class for pivot reversal scalping with gate filters."""
    
    def __init__(
        self,
        contract_qty: int = 4,
        daily_profit_target: float = 2000.0,
        daily_loss_limit: float = 350.0,
        use_session: bool = True,
        session_start: str = "09:30",
        session_end: str = "15:45",
        tz_str: str = "America/New_York",
        tp_ticks: int = 250,
        sl_ticks: int = 125,
        trail_pts: int = 20,
        use_atr_gate: bool = True,
        atr_min: float = 2.2,
        use_rsi_gate: bool = True,
        rsi_buy: int = 45,
        rsi_sell: int = 55,
        use_ema_gate: bool = False,
        ema_len: int = 9,
        use_vix_gate: bool = False,
        vix_max: float = 30.0,
        vix_min: float = 12.0,
        tick_size: float = 0.25,
        point_value: float = 2.0
    ):
        self.contract_qty = contract_qty
        self.daily_profit_target = daily_profit_target
        self.daily_loss_limit = daily_loss_limit
        self.use_session = use_session
        self.session_start = session_start
        self.session_end = session_end
        self.tz_str = tz_str
        self.tp_ticks = tp_ticks
        self.sl_ticks = sl_ticks
        self.trail_pts = trail_pts
        self.use_atr_gate = use_atr_gate
        self.atr_min = atr_min
        self.use_rsi_gate = use_rsi_gate
        self.rsi_buy = rsi_buy
        self.rsi_sell = rsi_sell
        self.use_ema_gate = use_ema_gate
        self.ema_len = ema_len
        self.use_vix_gate = use_vix_gate
        self.vix_max = vix_max
        self.vix_min = vix_min
        self.tick_size = tick_size
        self.point_value = point_value
        
        self._reset_daily_state()
        self._position_size = 0
        self._position_avg_price = 0.0
        self._entry_bar_idx = -1
        self._highest_price_since_entry = 0.0
        self._lowest_price_since_entry = 0.0
    
    def _reset_daily_state(self):
        """Reset daily P&L tracking."""
        self._day_eq = 0.0
        self._last_date = None
    
    def _check_date_reset(self, open_time: pd.Series) -> None:
        """Check if date changed and reset daily P&L."""
        current_date = open_time.dt.date
        if self._last_date is None:
            self._last_date = current_date.iloc[0] if len(current_date) > 0 else None
        elif len(current_date) > 0:
            if current_date.iloc[-1] != self._last_date:
                self._day_eq = 0.0
                self._last_date = current_date.iloc[-1]
    
    def _check_session(self, open_time: pd.Series) -> np.ndarray:
        """Check if bars are within session window."""
        if not self.use_session:
            return np.ones(len(open_time), dtype=bool)
        
        start_h, start_m = map(int, self.session_start.split(':'))
        end_h, end_m = map(int, self.session_end.split(':'))
        
        session_mask = np.zeros(len(open_time), dtype=bool)
        for i in range(len(open_time)):
            bar_time = open_time.iloc[i]
            if isinstance(bar_time, (datetime, pd.Timestamp)):
                bar_h, bar_m = bar_time.hour, bar_time.minute
                bar_minutes = bar_h * 60 + bar_m
                start_minutes = start_h * 60 + start_m
                end_minutes = end_h * 60 + end_m
                session_mask[i] = start_minutes <= bar_minutes <= end_minutes
        
        return session_mask
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> np.ndarray:
        """Calculate ATR."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.values
    
    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> np.ndarray:
        """Calculate RSI."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        return rsi.values
    
    def _calculate_ema(self, close: pd.Series, period: int) -> np.ndarray:
        """Calculate EMA."""
        return close.ewm(span=period, adjust=False).mean().values
    
    def _calculate_pivots(self, high: pd.Series, low: pd.Series, length: int = 1) -> tuple:
        """Calculate pivot highs and lows."""
        ph = np.zeros(len(high), dtype=bool)
        pl = np.zeros(len(low), dtype=bool)
        
        for i in range(length, len(high) - length):
            if i < length:
                continue
            is_ph = True
            is_pl = True
            for j in range(i - length, i + length + 1):
                if j == i:
                    continue
                if high.iloc[j] >= high.iloc[i]:
                    is_ph = False
                if low.iloc[j] <= low.iloc[i]:
                    is_pl = False
            if is_ph:
                ph[i] = True
            if is_pl:
                pl[i] = True
        
        return ph, pl
    
    def _check_vix_gate(self, vix_values: Optional[np.ndarray] = None) -> np.ndarray:
        """Check VIX gate (requires external data)."""
        if not self.use_vix_gate:
            return np.ones(len(vix_values) if vix_values is not None else 1, dtype=bool)
        
        if vix_values is None:
            return np.zeros(1, dtype=bool)
        
        return (vix_values <= self.vix_max) & (vix_values >= self.vix_min)
    
    def _update_position_state(self, close: np.ndarray, high: np.ndarray, low: np.ndarray, bar_idx: int) -> None:
        """Update trailing stop tracking."""
        if self._position_size > 0:
            self._highest_price_since_entry = max(self._highest_price_since_entry, high[bar_idx])
        elif self._position_size < 0:
            self._lowest_price_since_entry = min(self._lowest_price_since_entry, low[bar_idx])
    
    def _check_exit_conditions(self, close: np.ndarray, high: np.ndarray, low: np.ndarray, bar_idx: int) -> int:
        """Check if position should be exited. Returns 0=hold, 1=exit_long, -1=exit_short."""
        if self._position_size == 0:
            return 0
        
        entry_price = self._position_avg_price
        current_price = close[bar_idx]
        
        if self._position_size > 0:
            sl_price = entry_price - self.sl_ticks * self.tick_size
            tp_price = entry_price + self.tp_ticks * self.tick_size
            trail_price = self._highest_price_since_entry - self.trail_pts * self.tick_size
            
            if low[bar_idx] <= sl_price or high[bar_idx] >= tp_price or low[bar_idx] <= trail_price:
                return 1
        else:
            sl_price = entry_price + self.sl_ticks * self.tick_size
            tp_price = entry_price - self.tp_ticks * self.tick_size
            trail_price = self._lowest_price_since_entry + self.trail_pts * self.tick_size
            
            if high[bar_idx] >= sl_price or low[bar_idx] <= tp_price or high[bar_idx] >= trail_price:
                return -1
        
        return 0
    
    def _update_daily_pnl(self, close: np.ndarray, bar_idx: int, exit_pnl: float) -> None:
        """Update daily P&L tracking."""
        self._day_eq += exit_pnl
    
    def generate_signals(
        self,
        prices: pd.DataFrame,
        vix_values: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate trading signals.
        
        Args:
            prices: DataFrame with columns [open_time, open, high, low, close, volume]
            vix_values: Optional array of VIX values for VIX gate
        
        Returns:
            numpy.ndarray of position intent: 1=long, -1=short, 0=flat
        """
        n = len(prices)
        signals = np.zeros(n, dtype=np.int8)
        
        if n == 0:
            return signals
        
        open_time = prices['open_time']
        open_price = prices['open']
        high = prices['high']
        low = prices['low']
        close = prices['close']
        
        self._check_date_reset(open_time)
        
        atr = self._calculate_atr(high, low, close, 14)
        rsi = self._calculate_rsi(close, 14)
        ema = self._calculate_ema(close, self.ema_len) if self.use_ema_gate else None
        ph, pl = self._calculate_pivots(high, low, 1)
        
        session_mask = self._check_session(open_time)
        vix_mask = self._check_vix_gate(vix_values)
        
        daily_pnl = self._day_eq
        gate_daily = (daily_pnl < self.daily_profit_target) and (daily_pnl > -self.daily_loss_limit)
        
        for i in range(n):
            if i < 14:
                signals[i] = 0
                continue
            
            gate_session = session_mask[i]
            gate_atr = not self.use_atr_gate or atr[i] > self.atr_min
            gate_rsi_long = not self.use_rsi_gate or rsi[i] < self.rsi_buy
            gate_rsi_short = not self.use_rsi_gate or rsi[i] > self.rsi_sell
            gate_ema_long = not self.use_ema_gate or close.iloc[i] > ema[i]
            gate_ema_short = not self.use_ema_gate or close.iloc[i] < ema[i]
            gate_vix = vix_mask[i] if len(vix_mask) > i else True
            
            is_flat = self._position_size == 0
            
            if is_flat:
                go_long = (
                    gate_session and gate_daily and gate_vix and
                    pl[i] and gate_atr and gate_rsi_long and gate_ema_long
                )
                go_short = (
                    gate_session and gate_daily and gate_vix and
                    ph[i] and gate_atr and gate_rsi_short and gate_ema_short
                )
                
                if go_long:
                    signals[i] = 1
                    self._position_size = self.contract_qty
                    self._position_avg_price = close.iloc[i]
                    self._entry_bar_idx = i
                    self._highest_price_since_entry = close.iloc[i]
                elif go_short:
                    signals[i] = -1
                    self._position_size = -self.contract_qty
                    self._position_avg_price = close.iloc[i]
                    self._entry_bar_idx = i
                    self._lowest_price_since_entry = close.iloc[i]
            else:
                exit_signal = self._check_exit_conditions(close.values, high.values, low.values, i)
                if exit_signal != 0:
                    exit_pnl = 0.0
                    if self._position_size > 0:
                        exit_pnl = (close.iloc[i] - self._position_avg_price) * abs(self._position_size) * self.point_value
                    else:
                        exit_pnl = (self._position_avg_price - close.iloc[i]) * abs(self._position_size) * self.point_value
                    
                    self._update_daily_pnl(close.values, i, exit_pnl)
                    self._position_size = 0
                    self._position_avg_price = 0.0
                    self._entry_bar_idx = -1
                
                self._update_position_state(close.values, high.values, low.values, i)
                signals[i] = np.sign(self._position_size)
        
        return signals


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Standalone signal generation function for repo compatibility.
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume]
    
    Returns:
        numpy.ndarray of position intent: 1=long, -1=short, 0=flat
    """
    strategy = TriggerHappyElite()
    return strategy.generate_signals(prices)
