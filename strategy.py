# 12h Camarilla Pivot Breakout with Volume Confirmation and Trend Filter
# Targets 20-35 trades/year per symbol using 1d Camarilla levels (S3/R3) and 1w trend filter
# Breakouts occur at key pivot levels with volume confirmation to filter false breakouts
# Uses 1w EMA for trend direction to align with higher timeframe momentum
# Designed to work in both bull and bear markets by following the trend
# Expected trade frequency: 20-35/year per symbol to minimize fee drag

#!/usr/bin/env python3
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
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Typical price: (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Pivot point
    pivot = typical_price
    # Ranges
    range_hl = df_1d['high'] - df_1d['low']
    # Camarilla levels
    r3 = pivot + (range_hl * 1.1 / 2)
    r2 = pivot + (range_hl * 1.1 / 4)
    r1 = pivot + (range_hl * 1.1 / 6)
    s1 = pivot - (range_hl * 1.1 / 6)
    s2 = pivot - (range_hl * 1.1 / 4)
    s3 = pivot - (range_hl * 1.1 / 2)
    
    # Align to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, r3.values)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2.values)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1.values)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2.values)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3.values)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 34-period EMA on 1w close for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: volume > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Breakout logic with volume confirmation
        if close[i] > r3_12h[i] and volume_filter[i]:  # Break above R3
            if close[i] > ema34_1w_aligned[i]:  # Only long in uptrend
                signals[i] = 0.30
                position = 1
        elif close[i] < s3_12h[i] and volume_filter[i]:  # Break below S3
            if close[i] < ema34_1w_aligned[i]:  # Only short in downtrend
                signals[i] = -0.30
                position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_S3R3_Breakout_1wEMA34_VolumeFilter"
timeframe = "12h"
leverage = 1.0