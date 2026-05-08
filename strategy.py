#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Uses 1d price channel breakouts for trend following, filtered by weekly trend direction.
# Volume confirmation reduces false breakouts. Designed for low-frequency trades (<100 total)
# to minimize fee drift, targeting 30-100 trades over 4 years. Works in both bull and bear
# markets by only taking breakouts in the direction of the weekly trend.

name = "1d_Donchian20_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period Donchian channels (using previous 20 days)
    highest_high = np.full_like(high_1d, np.nan)
    lowest_low = np.full_like(low_1d, np.nan)
    for i in range(20, len(high_1d)):
        highest_high[i] = np.max(high_1d[i-20:i])
        lowest_low[i] = np.min(low_1d[i-20:i])
    
    # Align Donchian levels to 1d timeframe (already aligned, but using helper for consistency)
    highest_high_aligned = align_htf_to_ltf(prices, df_1d, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_1d, lowest_low)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike (1.5x 20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper Donchian with weekly uptrend and volume spike
            if (close[i] > highest_high_aligned[i] and 
                close[i] > ema34_1w_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian with weekly downtrend and volume spike
            elif (close[i] < lowest_low_aligned[i] and 
                  close[i] < ema34_1w_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian or trend fails
            if (close[i] < lowest_low_aligned[i] or 
                close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Donchian or trend fails
            if (close[i] > highest_high_aligned[i] or 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals