#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot levels with volume confirmation
# Fade at R3/S3 levels (mean reversion) and breakout continuation at R4/S4 levels (trend following)
# Volume confirmation filters false signals (current volume > 1.5x 20-period average)
# Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
# Works in bull/bear: mean reversion in ranging markets, breakout continuation in trending markets

name = "6h_1d_camarilla_pivot_breakout_fade_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + (Range * 1.1/2)
    # R3 = C + (Range * 1.1/4)
    # S3 = C - (Range * 1.1/4)
    # S4 = C - (Range * 1.1/2)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r4_1d = close_1d + (range_1d * 1.1 / 2.0)
    r3_1d = close_1d + (range_1d * 1.1 / 4.0)
    s3_1d = close_1d - (range_1d * 1.1 / 4.0)
    s4_1d = close_1d - (range_1d * 1.1 / 2.0)
    
    # Align 1d Camarilla levels to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 6h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit long if price reaches R4 (take profit) or breaks below S3 (stop loss)
            if close[i] >= r4_1d_aligned[i] or close[i] <= s3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price reaches S4 (take profit) or breaks above R3 (stop loss)
            if close[i] <= s4_1d_aligned[i] or close[i] >= r3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Fade strategy: enter short at R3, long at S3 with volume confirmation
            if close[i] >= r3_1d_aligned[i] and volume_confirmed:
                position = -1  # Short at R3 (fade)
                signals[i] = -0.25
            elif close[i] <= s3_1d_aligned[i] and volume_confirmed:
                position = 1   # Long at S3 (fade)
                signals[i] = 0.25
            # Breakout strategy: enter long at R4, short at S4 with volume confirmation
            elif close[i] > r4_1d_aligned[i] and volume_confirmed:
                position = 1   # Long breakout at R4
                signals[i] = 0.25
            elif close[i] < s4_1d_aligned[i] and volume_confirmed:
                position = -1  # Short breakout at S4
                signals[i] = -0.25
    
    return signals