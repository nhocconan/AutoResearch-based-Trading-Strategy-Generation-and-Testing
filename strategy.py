#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above 20-period high AND price > EMA50(1d) AND volume > 1.5x 20-period average.
# Short when price breaks below 20-period low AND price < EMA50(1d) AND volume > 1.5x 20-period average.
# Exit when price crosses back below the 20-period moving average (for long) or above it (for short).
# Donchian provides clear breakout levels. EMA50 filters trend direction on higher timeframe.
# Volume confirmation ensures institutional participation. Target: 50-150 total trades over 4 years.

name = "12h_Donchian_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    avg_roll = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # 12h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # EMA50 on 1d close
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(avg_roll[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Donchian high, price > EMA50, volume filter
            long_cond = (close[i] > high_roll[i]) and (close[i] > ema_50_aligned[i]) and volume_filter[i]
            # Short conditions: break below Donchian low, price < EMA50, volume filter
            short_cond = (close[i] < low_roll[i]) and (close[i] < ema_50_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: cross below 20-period average
            if close[i] < avg_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: cross above 20-period average
            if close[i] > avg_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals