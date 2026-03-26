#!/usr/bin/env python3
"""
Experiment #021: 4h Supertrend Breakout + 1d EMA + Volume

HYPOTHESIS: Supertrend is a cleaner, more responsive breakout signal than Donchian.
Combined with 1d EMA for trend alignment and volume confirmation, this captures
institutional breakout points. Works in both bull (long breakouts above 1d EMA)
and bear (short breakouts below 1d EMA with rallies to EMA as shorts).

Key changes from failed strategies:
- Supertrend instead of Donchian (more responsive, generates more trades)
- Volume spike at 1.5x (less strict than 2.0x)
- 1d EMA alignment (50 period - enough alignment but not too strict)
- Simple flip detection without complex cross-bars
- ATR-based trailing stoploss

TIMEFRAME: 4h primary
HTF: 1d for trend bias
TARGET: 75-200 total trades over 4 years (19-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_supertrend_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Supertrend indicator - returns trend direction (1=bull, -1=bear)"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n, dtype=np.int32)
    
    # ATR calculation
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    k = 1.0 / period
    atr = np.zeros(n, dtype=np.float64)
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = k * tr[i] + (1 - k) * atr[i - 1]
    
    # Upper and lower bands
    upper_band = (high + low) / 2.0 + multiplier * atr
    lower_band = (high + low) / 2.0 - multiplier * atr
    
    # Initial trend
    trend = np.full(n, 0, dtype=np.int32)
    trend[period - 1] = 1
    
    # Preliminary trend determination
    for i in range(period, n):
        if close[i] > upper_band[i - 1]:
            trend[i] = 1
        elif close[i] < lower_band[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
    
    # Final trend with band switches
    final_trend = np.full(n, 0, dtype=np.int32)
    final_trend[period - 1] = trend[period - 1]
    
    upper_band_final = upper_band.copy()
    lower_band_final = lower_band.copy()
    
    for i in range(period, n):
        if trend[i] == 1:
            lower_band_final[i] = max(lower_band_final[i], lower_band_final[i - 1])
            if close[i] < lower_band_final[i - 1]:
                final_trend[i] = -1
            else:
                final_trend[i] = 1
        else:
            upper_band_final[i] = min(upper_band_final[i], upper_band_final[i - 1])
            if close[i] > upper_band_final[i - 1]:
                final_trend[i] = 1
            else:
                final_trend[i] = -1
    
    return final_trend

def calculate_ema(data, period):
    """Exponential Moving Average"""
    return pd.Series(data).ewm(span=period, min_periods=period, adjust=False).mean().values


def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA for trend bias
    ema_1d_raw = calculate_ema(close_1d, period=50)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_raw)
    
    # Local 4h Supertrend
    supertrend = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    
    # Local ATR for stoploss
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 50
    MIN_HOLD_BARS = 3
    
    for i in range(warmup, n):
        if np.isnan(atr_14[i]) or np.isnan(ema_1d_aligned[i]) or vol_ma[i] <= 0:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Trend: price vs 1d EMA
        price_above_ema = close[i] > ema_1d_aligned[i]
        
        # Volume spike
        vol_spike = vol_ratio[i] > 1.5
        
        # Supertrend direction
        st_dir = supertrend[i]
        
        # Supertrend flip detection (simple: compare to previous bar)
        st_flip_up = (supertrend[i] == 1) and (supertrend[i - 1] == -1)
        st_flip_down = (supertrend[i] == -1) and (supertrend[i - 1] == 1)
        
        desired_signal = 0.0
        
        if not in_position:
            # New long: Supertrend flips bullish + price above 1d EMA + volume spike
            if st_flip_up and price_above_ema and vol_spike:
                desired_signal = SIZE
            
            # New short: Supertrend flips bearish + price below 1d EMA + volume spike
            if st_flip_down and not price_above_ema and vol_spike:
                desired_signal = -SIZE
        
        # Stoploss check (2.5 ATR trailing)
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # Exit: opposite Supertrend flip OR ATR stoploss
        exit_triggered = False
        
        if in_position and position_side > 0:
            if st_flip_down:
                exit_triggered = True
        
        if in_position and position_side < 0:
            if st_flip_up:
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
        
        # Update position tracking
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
            else:
                # Same direction - maintain position (no churn)
                pass
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                entry_bar = 0
        
        signals[i] = desired_signal
    
    return signals