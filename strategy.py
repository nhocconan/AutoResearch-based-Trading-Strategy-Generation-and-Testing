#!/usr/bin/env python3
# 6h_1d_1w_pivots_momentum_v1
# Hypothesis: Trade with the weekly trend using daily pivot points and volume confirmation on 6h.
# In weekly uptrend: go long when price breaks above R1 with volume.
# In weekly downtrend: go short when price breaks below S1 with volume.
# Exit when price reverses to pivot point or weekly trend changes.
# Uses weekly EMA20 for trend filter and volume > 1.5x 20-period average for confirmation.
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_pivots_momentum_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot points and support/resistance levels for previous day
    pivot = np.zeros(len(high_1d))
    r1 = np.zeros(len(high_1d))
    s1 = np.zeros(len(high_1d))
    r2 = np.zeros(len(high_1d))
    s2 = np.zeros(len(high_1d))
    
    for i in range(1, len(high_1d)):
        # Previous day's OHLC
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        
        # Standard pivot point calculation
        pivot[i] = (prev_high + prev_low + prev_close) / 3.0
        r1[i] = 2 * pivot[i] - prev_low
        s1[i] = 2 * pivot[i] - prev_high
        r2[i] = pivot[i] + (prev_high - prev_low)
        s2[i] = pivot[i] - (prev_high - prev_low)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price < pivot (reversal) or weekly trend breaks (price < weekly EMA20)
            if close[i] < pivot_aligned[i] or close[i] < ema20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > pivot (reversal) or weekly trend breaks (price > weekly EMA20)
            if close[i] > pivot_aligned[i] or close[i] > ema20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price > R1 with volume surge and weekly uptrend
            if (close[i] > r1_aligned[i] and vol_surge and 
                close[i] > ema20_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price < S1 with volume surge and weekly downtrend
            elif (close[i] < s1_aligned[i] and vol_surge and 
                  close[i] < ema20_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals