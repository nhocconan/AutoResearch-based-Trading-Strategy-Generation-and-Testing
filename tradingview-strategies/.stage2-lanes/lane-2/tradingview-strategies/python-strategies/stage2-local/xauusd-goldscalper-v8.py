"""
XAUUSD GoldScalper Pro v8+Session
Converted from TradingView Pine Script
Timeframe: 5-minute optimized for XAUUSD/Gold
"""

import numpy as np
import pandas as pd
from datetime import timezone, timedelta

name = "xauusd-goldscalper-v8"
timeframe = "5m"
leverage = 10


def _ema(series: np.ndarray, length: int) -> np.ndarray:
    """Calculate Exponential Moving Average."""
    result = np.full_like(series, np.nan, dtype=np.float64)
    multiplier = 2.0 / (length + 1.0)
    result[0] = series[0]
    for i in range(1, len(series)):
        if not np.isnan(series[i]):
            if np.isnan(result[i-1]):
                result[i] = series[i]
            else:
                result[i] = (series[i] - result[i-1]) * multiplier + result[i-1]
    return result


def _macd(close: np.ndarray, fast: int, slow: int, signal: int):
    """Calculate MACD line, signal line, and histogram."""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _rsi(close: np.ndarray, length: int) -> np.ndarray:
    """Calculate Relative Strength Index."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = _ema(gain, length)
    avg_loss = _ema(loss, length)
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = _ema(true_range, length)
    return atr


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    """Calculate ADX (Average Directional Index)."""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(low)
    
    for i in range(1, len(high)):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    atr = _atr(high, low, close, length)
    plus_di = 100.0 * _ema(plus_dm, length) / atr
    minus_di = 100.0 * _ema(minus_dm, length) / atr
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = _ema(dx, length)
    return adx


def _crossover(series1: np.ndarray, series2: np.ndarray) -> np.ndarray:
    """Detect crossover (series1 crosses above series2)."""
    result = np.zeros(len(series1), dtype=bool)
    for i in range(1, len(series1)):
        if not np.isnan(series1[i]) and not np.isnan(series2[i]):
            if not np.isnan(series1[i-1]) and not np.isnan(series2[i-1]):
                if series1[i-1] <= series2[i-1] and series1[i] > series2[i]:
                    result[i] = True
    return result


def _crossunder(series1: np.ndarray, series2: np.ndarray) -> np.ndarray:
    """Detect crossunder (series1 crosses below series2)."""
    result = np.zeros(len(series1), dtype=bool)
    for i in range(1, len(series1)):
        if not np.isnan(series1[i]) and not np.isnan(series2[i]):
            if not np.isnan(series1[i-1]) and not np.isnan(series2[i-1]):
                if series1[i-1] >= series2[i-1] and series1[i] < series2[i]:
                    result[i] = True
    return result


def _get_kst_hour(open_time: pd.Series) -> np.ndarray:
    """Extract hour in KST (UTC+9) from datetime index."""
    hours = np.zeros(len(open_time), dtype=np.int32)
    for i, ts in enumerate(open_time):
        if hasattr(ts, 'hour'):
            if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
                ts_utc = ts.astimezone(timezone.utc)
                ts_kst = ts_utc + timedelta(hours=9)
                hours[i] = ts_kst.hour
            else:
                hours[i] = (ts.hour + 9) % 24
        else:
            hours[i] = 0
    return hours


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Generate trading signals for XAUUSD GoldScalper Pro v8.
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume]
        
    Returns:
        numpy.ndarray of signals: 1=long entry, -1=short entry, 0=hold/exit
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.int8)
    
    if n < 100:
        return signals
    
    open_time = prices['open_time'].values
    open_p = prices['open'].values.astype(np.float64)
    high = prices['high'].values.astype(np.float64)
    low = prices['low'].values.astype(np.float64)
    close = prices['close'].values.astype(np.float64)
    
    e1, e2, e3 = 8, 21, 55
    e3_slope_bars = 10
    m1, m2, m3 = 12, 26, 9
    use_macd_zero = True
    r_len = 14
    r_long_min, r_long_max = 50, 68
    r_short_min, r_short_max = 32, 50
    a_len = 14
    sl_m, tp1_m, tp2_m = 1.0, 2.0, 5.0
    use_be = True
    adx_min = 20
    use_candle = True
    use_session = True
    kst_block_start, kst_block_end = 0, 6
    
    ema1 = _ema(close, e1)
    ema2 = _ema(close, e2)
    ema3 = _ema(close, e3)
    
    macd_line, signal_line, hist = _macd(close, m1, m2, m3)
    adx = _adx(high, low, close, 14)
    rsi = _rsi(close, r_len)
    atr = _atr(high, low, close, a_len)
    
    bull_align = (ema1 > ema2) & (ema2 > ema3)
    bear_align = (ema1 < ema2) & (ema2 < ema3)
    
    ema3_lookback = np.roll(ema3, e3_slope_bars)
    ema3_lookback[:e3_slope_bars] = ema3[0]
    ema3_up = ema3 > ema3_lookback
    ema3_down = ema3 < ema3_lookback
    
    macd_up = _crossover(macd_line, signal_line)
    macd_down = _crossunder(macd_line, signal_line)
    
    macd_sma20 = _ema(macd_line, 20)
    macd_zero_long = (~use_macd_zero) | (macd_line > 0) | ((macd_line > np.roll(macd_line, 1)) & (macd_line > -np.abs(macd_sma20)))
    macd_zero_short = (~use_macd_zero) | (macd_line < 0) | ((macd_line < np.roll(macd_line, 1)) & (macd_line < np.abs(macd_sma20)))
    macd_zero_long[:1] = False
    macd_zero_short[:1] = False
    
    rsi_long_ok = (rsi >= r_long_min) & (rsi <= r_long_max)
    rsi_short_ok = (rsi <= r_short_max) & (rsi >= r_short_min)
    
    adx_ok = adx >= adx_min
    
    bull_candle = (~use_candle) | (close > open_p)
    bear_candle = (~use_candle) | (close < open_p)
    
    kst_hours = _get_kst_hour(pd.Series(open_time))
    in_block = (kst_hours >= kst_block_start) & (kst_hours < kst_block_end)
    session_ok = (~use_session) | (~in_block)
    
    in_long = np.zeros(n, dtype=bool)
    in_short = np.zeros(n, dtype=bool)
    entry_px = np.zeros(n, dtype=np.float64)
    be_active = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        long_sig = (macd_up[i] and bull_align[i] and ema3_up[i] and 
                   rsi_long_ok[i] and adx_ok[i] and macd_zero_long[i] and 
                   bull_candle[i] and session_ok[i] and not in_long[i-1] and not in_short[i-1])
        
        short_sig = (macd_down[i] and bear_align[i] and ema3_down[i] and 
                    rsi_short_ok[i] and adx_ok[i] and macd_zero_short[i] and 
                    bear_candle[i] and session_ok[i] and not in_long[i-1] and not in_short[i-1])
        
        if long_sig:
            in_long[i] = True
            entry_px[i] = close[i]
            signals[i] = 1
        elif short_sig:
            in_short[i] = True
            entry_px[i] = close[i]
            signals[i] = -1
        else:
            in_long[i] = in_long[i-1]
            in_short[i] = in_short[i-1]
            entry_px[i] = entry_px[i-1]
        
        if in_long[i-1]:
            tp1_hit = high[i] >= entry_px[i-1] + atr[i-1] * tp1_m
            if tp1_hit and not be_active[i-1]:
                be_active[i] = True
            else:
                be_active[i] = be_active[i-1]
            
            long_exit = _crossunder(ema1, ema2)[i] or rsi[i] > r_long_max + 5
            be_exit = use_be and be_active[i] and low[i] < entry_px[i-1] + atr[i-1] * 0.1
            sl_exit = low[i] < entry_px[i-1] - atr[i-1] * sl_m
            tp2_exit = high[i] >= entry_px[i-1] + atr[i-1] * tp2_m
            
            if long_exit or be_exit or sl_exit or tp2_exit:
                in_long[i] = False
                be_active[i] = False
                if signals[i] == 0:
                    signals[i] = 0
        else:
            be_active[i] = be_active[i-1] if i > 0 else False
        
        if in_short[i-1]:
            tp1_hit = low[i] <= entry_px[i-1] - atr[i-1] * tp1_m
            if tp1_hit and not be_active[i-1]:
                be_active[i] = True
            else:
                be_active[i] = be_active[i-1]
            
            short_exit = _crossover(ema1, ema2)[i] or rsi[i] < r_short_min - 5
            be_exit = use_be and be_active[i] and high[i] > entry_px[i-1] - atr[i-1] * 0.1
            sl_exit = high[i] > entry_px[i-1] + atr[i-1] * sl_m
            tp2_exit = low[i] <= entry_px[i-1] - atr[i-1] * tp2_m
            
            if short_exit or be_exit or sl_exit or tp2_exit:
                in_short[i] = False
                be_active[i] = False
        
        if i == 0:
            be_active[i] = False
    
    return signals
