# 6h_1w_PivotBreakout_VolumeFilter_v1
# Hypothesis: Weekly pivot points from 1w timeframe act as significant support/resistance levels. 
# Price breaking above weekly R1 with volume confirmation indicates bullish momentum continuation.
# Price breaking below weekly S1 with volume confirmation indicates bearish momentum continuation.
# Weekly timeframe filters out noise and works in both bull/bear markets by following institutional levels.
# Volume confirmation ensures breakouts are genuine, reducing false signals.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25 to minimize churn.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w HTF data once for weekly pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot levels (standard formula)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pw = (high_1w + low_1w + close_1w) / 3.0
    r1w = 2 * pw - low_1w
    s1w = 2 * pw - high_1w
    r2w = pw + (high_1w - low_1w)
    s2w = pw - (high_1w - low_1w)
    
    # Align all 1w indicators to 6h timeframe
    pw_aligned = align_htf_to_ltf(prices, df_1w, pw)
    r1w_aligned = align_htf_to_ltf(prices, df_1w, r1w)
    s1w_aligned = align_htf_to_ltf(prices, df_1w, s1w)
    r2w_aligned = align_htf_to_ltf(prices, df_1w, r2w)
    s2w_aligned = align_htf_to_ltf(prices, df_1w, s2w)
    
    # Main timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 2.0x 24-period average (6h * 4 = 1 day)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_filter = volume / np.where(vol_ma_24 == 0, 1, vol_ma_24) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(pw_aligned[i]) or np.isnan(r1w_aligned[i]) or np.isnan(s1w_aligned[i]) or
            np.isnan(r2w_aligned[i]) or np.isnan(s2w_aligned[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        high_i = high[i]
        low_i = low[i]
        pw_val = pw_aligned[i]
        r1w_val = r1w_aligned[i]
        s1w_val = s1w_aligned[i]
        r2w_val = r2w_aligned[i]
        s2w_val = s2w_aligned[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: price breaks above weekly R1 with volume (breakout continuation)
            if high_i > r1w_val and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with volume (breakdown continuation)
            elif low_i < s1w_val and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below weekly pivot OR volume filter fails
            if low_i < pw_val or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above weekly pivot OR volume filter fails
            if high_i > pw_val or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1w_PivotBreakout_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0