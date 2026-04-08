#!/usr/bin/env python3
# 6h_1d_ma_crossover_volume_filter_v1
# Hypothesis: Trade MA crossovers on 6h timeframe with volume confirmation and 1d trend filter.
# Uses 6h EMA8/EMA21 crossover for entry, confirmed by volume > 1.5x 20-period average.
# Trend filter: only take longs when 6h price > 1d EMA50, shorts when 6h price < 1d EMA50.
# Designed for 6h timeframe to target 12-37 trades/year (50-150 total over 4 years).
# Works in both bull and bear markets by aligning with higher timeframe trend and requiring volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ma_crossover_volume_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA50 for 1d trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 6h EMA8 and EMA21 for crossover
    ema8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure EMA21 and 1d EMA50 are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: EMA8 crosses below EMA21
            if ema8[i] < ema21[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: EMA8 crosses above EMA21
            if ema8[i] > ema21[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: EMA8 crosses above EMA21 with volume surge and 1d uptrend
            if (ema8[i] > ema21[i] and ema8[i-1] <= ema21[i-1] and vol_surge and 
                close[i] > ema50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: EMA8 crosses below EMA21 with volume surge and 1d downtrend
            elif (ema8[i] < ema21[i] and ema8[i-1] >= ema21[i-1] and vol_surge and 
                  close[i] < ema50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals