#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume spike.
Long when price breaks above Donchian upper AND price > 1w EMA50 AND volume > 2.0x 24-period average.
Short when price breaks below Donchian lower AND price < 1w EMA50 AND volume > 2.0x 24-period average.
Exit when price reverts to Donchian midpoint OR ATR trailing stop (2.5*ATR from extreme).
Uses 1w HTF for trend alignment and Donchian levels from 1d.
Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian levels from previous day (using daily data)
    # Use previous day's data to avoid look-ahead
    high_1d = get_htf_data(prices, '1d')
    if len(high_1d) < 20:  # Need enough for Donchian
        return np.zeros(n)
    
    high_1d_prev = np.roll(high_1d['high'].values, 1)
    low_1d_prev = np.roll(high_1d['low'].values, 1)
    # First value will be NaN due to roll, handled by min_periods in align
    
    # Donchian(20) from previous day
    upper_20 = pd.Series(high_1d_prev).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d_prev).rolling(window=20, min_periods=20).min().values
    mid_20 = (upper_20 + lower_20) / 2.0
    
    # Align Donchian levels to 12h timeframe (use previous day's levels)
    upper_aligned = align_htf_to_ltf(prices, high_1d, upper_20)
    lower_aligned = align_htf_to_ltf(prices, high_1d, lower_20)
    mid_aligned = align_htf_to_ltf(prices, high_1d, mid_20)
    
    # 12h volume average (24-period = 12 days) for spike filter
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # ATR(14) for 12h trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(24, 50, 20, 1)  # vol_ma24, ema_50_1w, donchian20, and +1 for roll
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(mid_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_50_1w_aligned[i]
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        mid_val = mid_aligned[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above upper AND price > 1w EMA50 AND volume spike
            if price > upper_val and price > ema_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: price breaks below lower AND price < 1w EMA50 AND volume spike
            elif price < lower_val and price < ema_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price reverts to midpoint
            if position == 1 and price < mid_val:
                exit_signal = True
            elif position == -1 and price > mid_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_MidExit_ATRTrailingStop"
timeframe = "12h"
leverage = 1.0