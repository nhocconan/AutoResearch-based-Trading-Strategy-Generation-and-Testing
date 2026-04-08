#!/usr/bin/env python3
# 6h_1d_camarilla_pivot_volume_v1
# Hypothesis: Use 1d Camarilla pivot levels for institutional support/resistance. Fade at R3/S3 (mean reversion), breakout continuation at R4/S4. Volume confirms institutional participation. Works in range (fade) and trend (breakout) markets.
# Target: 50-150 total trades over 4 years (12-37/year) with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_camarilla_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    # Based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and range
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r4 = close_1d + range_1d * 1.1 / 2
    r3 = close_1d + range_1d * 1.1 / 4
    s3 = close_1d - range_1d * 1.1 / 4
    s4 = close_1d - range_1d * 1.1 / 2
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: volume > 1.5x average of last 24 periods (1 day in 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or \
           np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S3 (mean reversion failure) or breaks above R4 (take profit)
            if close[i] < s3_aligned[i] or close[i] > r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above R3 (mean reversion failure) or breaks below S4 (take profit)
            if close[i] > r3_aligned[i] or close[i] < s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price rejects S3 with volume (mean reversion bounce)
            if (close[i] > s3_aligned[i] and 
                close[i] < s4_aligned[i] and  # Between S3 and S4
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price rejects R3 with volume (mean reversion rejection)
            elif (close[i] < r3_aligned[i] and 
                  close[i] > r4_aligned[i] and  # Between R3 and R4
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
            # Long breakout: price breaks above R4 with volume (continuation)
            elif close[i] > r4_aligned[i] and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short breakdown: price breaks below S4 with volume (continuation)
            elif close[i] < s4_aligned[i] and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals