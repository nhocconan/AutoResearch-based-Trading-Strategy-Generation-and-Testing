import numpy as np
import pandas as pd
from typing import Optional

name = "SMC + VP Pro Strategy [MR.M]"
timeframe = "1d"
leverage = 1.0

def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean().values
    rs = np.where(avg_loss == 0, 100.0, avg_gain / avg_loss)
    return 100.0 - (100.0 / (1.0 + rs))

def _pivot_high(high: np.ndarray, left: int, right: int) -> np.ndarray:
    result = np.full(len(high), np.nan)
    for i in range(left, len(high) - right):
        if np.all(high[i-left:i] < high[i]) and np.all(high[i+1:i+1+right] <= high[i]):
            result[i] = high[i]
    return result

def _pivot_low(low: np.ndarray, left: int, right: int) -> np.ndarray:
    result = np.full(len(low), np.nan)
    for i in range(left, len(low) - right):
        if np.all(low[i-left:i] > low[i]) and np.all(low[i+1:i+1+right] >= low[i]):
            result[i] = low[i]
    return result

def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    return pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values

def _detect_fvg(high: np.ndarray, low: np.ndarray, close: np.ndarray, min_size: float = 0.0001) -> tuple:
    n = len(close)
    bull_fvg_bottom = np.full(n, np.nan)
    bear_fvg_top = np.full(n, np.nan)
    for i in range(2, n):
        if low[i] > high[i-2] and close[i-1] > high[i-2]:
            size = low[i] - high[i-2]
            if size >= min_size:
                bull_fvg_bottom[i] = high[i-2]
        if high[i] < low[i-2] and close[i-1] < low[i-2]:
            size = low[i-2] - high[i]
            if size >= min_size:
                bear_fvg_top[i] = low[i-2]
    return bull_fvg_bottom, bear_fvg_top

def _detect_sweep(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                  swing_high: np.ndarray, swing_low: np.ndarray) -> tuple:
    n = len(close)
    bull_sweep = np.zeros(n, dtype=bool)
    bear_sweep = np.zeros(n, dtype=bool)
    last_sh = np.nan
    last_sl = np.nan
    for i in range(n):
        if not np.isnan(swing_high[i]):
            last_sh = swing_high[i]
        if not np.isnan(swing_low[i]):
            last_sl = swing_low[i]
        if not np.isnan(last_sh) and high[i] > last_sh and close[i] < last_sh:
            bear_sweep[i] = True
        if not np.isnan(last_sl) and low[i] < last_sl and close[i] > last_sl:
            bull_sweep[i] = True
    return bull_sweep, bear_sweep

def _detect_bos(close: np.ndarray, swing_high: np.ndarray, swing_low: np.ndarray) -> tuple:
    n = len(close)
    bull_bos = np.zeros(n, dtype=bool)
    bear_bos = np.zeros(n, dtype=bool)
    last_sh = np.nan
    last_sl = np.nan
    for i in range(n):
        if not np.isnan(swing_high[i]):
            last_sh = swing_high[i]
        if not np.isnan(swing_low[i]):
            last_sl = swing_low[i]
        if not np.isnan(last_sh) and close[i] > last_sh and close[i-1] <= last_sh:
            bull_bos[i] = True
        if not np.isnan(last_sl) and close[i] < last_sl and close[i-1] >= last_sl:
            bear_bos[i] = True
    return bull_bos, bear_bos

def _detect_divergence(high: np.ndarray, low: np.ndarray, rsi: np.ndarray, 
                       div_lookback: int = 5) -> tuple:
    n = len(close) if 'close' in dir() else len(high)
    n = len(high)
    bull_reg_div = np.zeros(n, dtype=bool)
    bear_reg_div = np.zeros(n, dtype=bool)
    bull_hid_div = np.zeros(n, dtype=bool)
    bear_hid_div = np.zeros(n, dtype=bool)
    
    price_pivot_low = np.full(n, np.nan)
    price_pivot_high = np.full(n, np.nan)
    rsi_pivot_low = np.full(n, np.nan)
    rsi_pivot_high = np.full(n, np.nan)
    
    ph = _pivot_high(high, div_lookback, div_lookback)
    pl = _pivot_low(low, div_lookback, div_lookback)
    rph = _pivot_high(rsi, div_lookback, div_lookback)
    rpl = _pivot_low(rsi, div_lookback, div_lookback)
    
    last_ppl = np.nan
    last_pph = np.nan
    last_rpl = np.nan
    last_rph = np.nan
    
    for i in range(n):
        if not np.isnan(pl[i]):
            if not np.isnan(last_ppl) and not np.isnan(last_rpl):
                cur_pl = low[i]
                cur_rl = rsi[i]
                if cur_pl < last_ppl and cur_rl > last_rpl:
                    bull_reg_div[i] = True
                elif cur_pl > last_ppl and cur_rl < last_rpl:
                    bull_hid_div[i] = True
            last_ppl = low[i]
            last_rpl = rsi[i] if not np.isnan(rpl[i]) else last_rpl
        if not np.isnan(ph[i]):
            if not np.isnan(last_pph) and not np.isnan(last_rph):
                cur_ph = high[i]
                cur_rh = rsi[i]
                if cur_ph > last_pph and cur_rh < last_rph:
                    bear_reg_div[i] = True
                elif cur_ph < last_pph and cur_rh > last_rph:
                    bear_hid_div[i] = True
            last_pph = high[i]
            last_rph = rsi[i] if not np.isnan(rph[i]) else last_rph
    
    return bull_reg_div, bear_reg_div, bull_hid_div, bear_hid_div

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Generate trading signals from OHLCV data.
    Returns: numpy array of -1 (short), 0 (neutral), 1 (long)
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.int8)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI
    rsi_period = 14
    rsi_overbought = 70.0
    rsi_oversold = 30.0
    rsi = _rsi(close, rsi_period)
    
    # Volume Profile proxy (VWAP-based POC approximation)
    typical_price = (high + low + close) / 3.0
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    vwap = np.where(np.isnan(vwap), close, vwap)
    poc_real_zone = np.abs(close - vwap) < (close * 0.005)
    
    # FVG detection
    fvg_min_size = 0.0001
    bull_fvg_bottom, bear_fvg_top = _detect_fvg(high, low, close, fvg_min_size)
    
    # Track active FVG levels
    active_bull_fvg_bottom = np.full(n, np.nan)
    active_bear_fvg_top = np.full(n, np.nan)
    for i in range(n):
        if i == 0:
            active_bull_fvg_bottom[i] = bull_fvg_bottom[i]
            active_bear_fvg_top[i] = bear_fvg_top[i]
        else:
            active_bull_fvg_bottom[i] = bull_fvg_bottom[i] if not np.isnan(bull_fvg_bottom[i]) else active_bull_fvg_bottom[i-1]
            active_bear_fvg_top[i] = bear_fvg_top[i] if not np.isnan(bear_fvg_top[i]) else active_bear_fvg_top[i-1]
    
    # OTE zones (62%-79% of FVG)
    in_bull_ote = np.zeros(n, dtype=bool)
    in_bear_ote = np.zeros(n, dtype=bool)
    for i in range(n):
        if not np.isnan(active_bull_fvg_bottom[i]):
            fvg_size = close[i] - active_bull_fvg_bottom[i] if close[i] > active_bull_fvg_bottom[i] else 0
            ote62 = active_bull_fvg_bottom[i] + fvg_size * 0.62
            ote79 = active_bull_fvg_bottom[i] + fvg_size * 0.79
            in_bull_ote[i] = ote62 <= close[i] <= ote79
        if not np.isnan(active_bear_fvg_top[i]):
            fvg_size = active_bear_fvg_top[i] - close[i] if close[i] < active_bear_fvg_top[i] else 0
            ote62 = active_bear_fvg_top[i] - fvg_size * 0.62
            ote79 = active_bear_fvg_top[i] - fvg_size * 0.79
            in_bear_ote[i] = ote79 <= close[i] <= ote62
    
    # Swing detection for BOS and liquidity
    swing_length = 5
    swing_high = _pivot_high(high, swing_length, swing_length)
    swing_low = _pivot_low(low, swing_length, swing_length)
    
    # BOS detection
    bull_bos, bear_bos = _detect_bos(close, swing_high, swing_low)
    
    # Liquidity sweep detection
    bull_sweep, bear_sweep = _detect_sweep(high, low, close, swing_high, swing_low)
    
    # Divergence detection
    div_lookback = 5
    bull_reg_div, bear_reg_div, bull_hid_div, bear_hid_div = _detect_divergence(
        high, low, rsi, div_lookback)
    
    # Signal mode (Conservative default)
    signal_mode = "Conservative"
    
    # Generate signals
    for i in range(n):
        near_real_zone = poc_real_zone[i]
        has_bull_fvg = not np.isnan(active_bull_fvg_bottom[i])
        has_bear_fvg = not np.isnan(active_bear_fvg_top[i])
        
        buy_signal = False
        sell_signal = False
        
        if signal_mode == "Conservative":
            buy_signal = bull_sweep[i] and (in_bull_ote[i] or near_real_zone or has_bull_fvg)
            sell_signal = bear_sweep[i] and (in_bear_ote[i] or near_real_zone or has_bear_fvg)
        elif signal_mode == "Balanced":
            buy_signal = bull_sweep[i] or (bull_bos[i] and has_bull_fvg) or (in_bull_ote[i] and bull_bos[i])
            sell_signal = bear_sweep[i] or (bear_bos[i] and has_bear_fvg) or (in_bear_ote[i] and bear_bos[i])
        else:  # Aggressive
            buy_signal = bull_sweep[i] or bull_bos[i] or in_bull_ote[i] or (has_bull_fvg and near_real_zone)
            sell_signal = bear_sweep[i] or bear_bos[i] or in_bear_ote[i] or (has_bear_fvg and near_real_zone)
        
        if buy_signal:
            signals[i] = 1
        elif sell_signal:
            signals[i] = -1
    
    return signals

if __name__ == "__main__":
    # Example usage
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    prices = pd.DataFrame({
        'open_time': dates,
        'open': 40000 + np.cumsum(np.random.randn(100) * 500),
        'high': 40000 + np.cumsum(np.random.randn(100) * 500) + np.random.rand(100) * 200,
        'low': 40000 + np.cumsum(np.random.randn(100) * 500) - np.random.rand(100) * 200,
        'close': 40000 + np.cumsum(np.random.randn(100) * 500),
        'volume': np.random.randint(1000, 10000, 100)
    })
    signals = generate_signals(prices)
    print(f"Signals generated: {len(signals)} bars")
    print(f"Long signals: {np.sum(signals == 1)}")
    print(f"Short signals: {np.sum(signals == -1)}")
