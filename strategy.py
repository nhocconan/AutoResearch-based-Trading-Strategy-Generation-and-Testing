# 12h_1d_Camarilla_Pivot_Volume_Strategy_v1
# Strategy: Use 1d Camarilla pivot levels as support/resistance with volume confirmation
# Long when price touches S3 with volume spike, short when price touches R3 with volume spike
# Works in both bull and bear markets by fading extreme moves at key pivot levels
# Low turnover expected: ~15-30 trades/year per symbol

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels (using previous day's data)
    # Camarilla levels based on previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation (shift by 1)
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    
    # Pivot point and range
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_hl * 1.1 / 2)
    s3 = pivot - (range_hl * 1.1 / 2)
    r4 = pivot + (range_hl * 1.1)
    s4 = pivot - (range_hl * 1.1)
    
    # Align 1d levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Enter long: price touches or goes below S3 with volume spike
            if (low[i] <= s3_aligned[i] and volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: price touches or goes above R3 with volume spike
            elif (high[i] >= r3_aligned[i] and volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches midpoint or S4 level
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2
            if close[i] >= midpoint or low[i] <= s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches midpoint or R4 level
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2
            if close[i] <= midpoint or high[i] >= r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Camarilla_Pivot_Volume_Strategy_v1"
timeframe = "12h"
leverage = 1.0