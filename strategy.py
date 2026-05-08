#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h trend filter (EMA50) and volume confirmation.
# Long when price breaks above 20-period high AND price > 4h EMA50 AND volume > 1.5x 20-period avg.
# Short when price breaks below 20-period low AND price < 4h EMA50 AND volume > 1.5x 20-period avg.
# Exit when price crosses back below 20-period high (long) or above 20-period low (short).
# Uses 4h EMA50 for trend direction to reduce false breakouts.
# Volume filter ensures institutional participation.
# Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag.

name = "1h_Donchian_20_4hEMA50_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # EMA50 on 4h close
    ema_50 = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Donchian channels (20-period high/low)
    high_max20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(high_max20[i]) or 
            np.isnan(low_min20[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above 20-period high, price > 4h EMA50, volume filter
            long_cond = (close[i] > high_max20[i]) and (close[i] > ema_50_aligned[i]) and volume_filter[i]
            # Short conditions: break below 20-period low, price < 4h EMA50, volume filter
            short_cond = (close[i] < low_min20[i]) and (close[i] < ema_50_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: cross below 20-period high
            if close[i] < high_max20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: cross above 20-period low
            if close[i] > low_min20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals