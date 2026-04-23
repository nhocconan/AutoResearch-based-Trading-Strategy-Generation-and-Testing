#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above 20-day high AND price > 1w EMA50 AND volume > 1.5x 20-day average.
Short when price breaks below 20-day low AND price < 1w EMA50 AND volume > 1.5x 20-day average.
Exit when price reverts to 10-day midpoint OR ATR trailing stop (2.0*ATR from extreme).
Uses 1w HTF for trend alignment and daily price structure.
Target: ~10-20 trades/year on 1d timeframe with discrete sizing 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Calculate Donchian channels from previous day (using daily data)
    # Use previous day's data to avoid look-ahead
    high_1d = pd.Series(high).rolling(window=1, min_periods=1).values  # daily equivalent
    low_1d = pd.Series(low).rolling(window=1, min_periods=1).values
    
    # For daily timeframe, we need to look back 20 days
    # Since we're on 1d timeframe, we can use rolling window directly
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align to previous day's values to avoid look-ahead
    highest_20_prev = np.roll(highest_20, 1)
    lowest_20_prev = np.roll(lowest_20, 1)
    
    # 10-day midpoint for exit
    midpoint_10 = (pd.Series(high).rolling(window=10, min_periods=10).max().values + 
                   pd.Series(low).rolling(window=10, min_periods=10).min().values) / 2.0
    midpoint_10_prev = np.roll(midpoint_10, 1)
    
    # 20-day volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
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
    start_idx = max(20, 50, 1)  # donchian20, ema_50_1w, and +1 for roll
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(highest_20_prev[i]) or np.isnan(lowest_20_prev[i]) or np.isnan(midpoint_10_prev[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_50_1w_aligned[i]
        highest_val = highest_20_prev[i]
        lowest_val = lowest_20_prev[i]
        midpoint_val = midpoint_10_prev[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above 20-day high AND price > 1w EMA50 AND volume spike
            if price > highest_val and price > ema_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: price breaks below 20-day low AND price < 1w EMA50 AND volume spike
            elif price < lowest_val and price < ema_val and volume[i] > 1.5 * vol_ma_val:
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
            
            # Primary exit: price reverts to 10-day midpoint
            if position == 1 and price < midpoint_val:
                exit_signal = True
            elif position == -1 and price > midpoint_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_MidpointExit_ATRTrailingStop"
timeframe = "1d"
leverage = 1.0