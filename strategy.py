# 4h_Camarilla_R1S1_With_Volume_Spike_and_Chop_Filter
# Hypothesis: Camarilla pivot levels (R1/S1) from daily chart act as strong support/resistance.
# In ranging markets (Chop > 61.8), price tends to reverse at these levels.
# In trending markets (Chop < 38.2), breakouts through R1/S1 continue the trend.
# Volume spike confirms institutional participation. Works in both bull/bear via regime filter.
# Target: 20-40 trades/year per symbol (~80-160 over 4 years).

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1S1_With_Volume_Spike_and_Chop_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # H, L, C = previous day's high, low, close
    H = np.concatenate([[np.nan], high_1d[:-1]])  # shift by 1
    L = np.concatenate([[np.nan], low_1d[:-1]])
    C = np.concatenate([[np.nan], close_1d[:-1]])
    
    # Camarilla R1, S1
    R1 = C + (H - L) * 1.1 / 12
    S1 = C - (H - L) * 1.1 / 12
    
    # Align to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Chopiness Index (14-period) for regime filter
    def true_range(high, low, close):
        tr1 = high - low
        tr2 = np.abs(high - np.concatenate([[np.nan], close[:-1]]))
        tr3 = np.abs(low - np.concatenate([[np.nan], close[:-1]]))
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr = true_range(high, low, close)
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Chop: log(sum(TR14) / (max(high14) - min(low14))) * 100 / log(14)
    tr14_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where(
        (max_high14 - min_low14) > 0,
        np.log10(tr14_sum / (max_high14 - min_low14)) * 100 / np.log10(14),
        50  # default if range is zero
    )
    
    # Volume spike: 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(chop[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions:
            # 1. Price crosses above S1 (support) from below
            # 2. Volume spike > 2.0
            # 3. Chop > 61.8 (ranging market) for mean reversion
            if (close[i] > S1_aligned[i] and close[i-1] <= S1_aligned[i-1] and
                volume_spike[i] > 2.0 and chop[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Short conditions:
            # 1. Price crosses below R1 (resistance) from above
            # 2. Volume spike > 2.0
            # 3. Chop > 61.8 (ranging market) for mean reversion
            elif (close[i] < R1_aligned[i] and close[i-1] >= R1_aligned[i-1] and
                  volume_spike[i] > 2.0 and chop[i] > 61.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches R1 or chop drops below 38.2 (trending) or volume drops
            if (close[i] >= R1_aligned[i] or chop[i] < 38.2 or volume_spike[i] < 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches S1 or chop drops below 38.2 or volume drops
            if (close[i] <= S1_aligned[i] or chop[i] < 38.2 or volume_spike[i] < 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals