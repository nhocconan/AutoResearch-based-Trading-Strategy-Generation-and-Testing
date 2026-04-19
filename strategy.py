#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 1d, with fade at R3/S3 and breakout at R4/S4
# Uses 1d Camarilla levels (H5, L5, H4, L4, H3, L3, H2, L2, H1, L1) calculated from prior day's range
# Entry logic: Long when price breaks above H4 with volume confirmation, short when breaks below L4
# Exit logic: Reverse position when price returns to H3/L3 levels or breaks H5/L5 for continuation
# Volume filter: requires volume > 1.5x 20-period average to avoid false breakouts
# Timeframe: 6h (primary), HTF: 1d for Camarilla calculation
# Target: 15-25 trades/year per symbol with controlled frequency
# Works in both bull/bear: breakouts capture trends, fades at R3/S3 capture reversals in range markets
name = "6h_Camarilla_R3S4_Breakout_Fade_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation (prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior day's range
    # Using prior day's high, low, close
    phigh = df_1d['high'].shift(1).values  # Prior day high
    plow = df_1d['low'].shift(1).values    # Prior day low
    pclose = df_1d['close'].shift(1).values # Prior day close
    
    # Calculate Camarilla levels
    H5 = pclose + 1.5 * (phigh - plow)
    H4 = pclose + 1.25 * (phigh - plow)
    H3 = pclose + 1.1 * (phigh - plow)
    L3 = pclose - 1.1 * (phigh - plow)
    L4 = pclose - 1.25 * (phigh - plow)
    L5 = pclose - 1.5 * (phigh - plow)
    
    # Align Camarilla levels to 6h timeframe (using prior day's levels for current day)
    H5_6h = align_htf_to_ltf(prices, df_1d, H5)
    H4_6h = align_htf_to_ltf(prices, df_1d, H4)
    H3_6h = align_htf_to_ltf(prices, df_1d, H3)
    L3_6h = align_htf_to_ltf(prices, df_1d, L3)
    L4_6h = align_htf_to_ltf(prices, df_1d, L4)
    L5_6h = align_htf_to_ltf(prices, df_1d, L5)
    
    # Volume filter: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(H4_6h[i]) or np.isnan(L4_6h[i]) or np.isnan(H3_6h[i]) or 
            np.isnan(L3_6h[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above H4 with volume
            if close[i] > H4_6h[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below L4 with volume
            elif close[i] < L4_6h[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price returns to H3 (fade) or breaks H5 (continuation - reverse)
            if close[i] < H3_6h[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] > H5_6h[i]:
                signals[i] = -0.25  # Reverse to short
                position = -1
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price returns to L3 (fade) or breaks L5 (continuation - reverse)
            if close[i] > L3_6h[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] < L5_6h[i]:
                signals[i] = 0.25  # Reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals