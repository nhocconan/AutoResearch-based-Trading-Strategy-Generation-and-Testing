#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal from 1d levels with volume confirmation.
# Uses 1d Camarilla levels (H4/L4 for breakouts, H3/L3 for reversals) as key support/resistance.
# Long when price pulls back to L3 with volume spike and closes above it.
# Short when price pulls back to H3 with volume spike and closes below it.
# Works in both bull and bear markets by fading extremes at proven institutional levels.
# Target: 50-150 total trades over 4 years (12-37/year) with strict entry filters.

name = "6h_camarilla1d_vol_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations (based on previous day)
    rng = high_1d - low_1d
    # H4 = close + 1.5 * rng * 1.1
    # L4 = close - 1.5 * rng * 1.1
    # H3 = close + 1.1 * rng * 1.1
    # L3 = close - 1.1 * rng * 1.1
    H4 = close_1d + 1.5 * rng * 1.1
    L4 = close_1d - 1.5 * rng * 1.1
    H3 = close_1d + 1.1 * rng * 1.1
    L3 = close_1d - 1.1 * rng * 1.1
    
    # Align levels to 6h timeframe
    H4_6h = align_htf_to_ltf(prices, df_1d, H4)
    L4_6h = align_htf_to_ltf(prices, df_1d, L4)
    H3_6h = align_htf_to_ltf(prices, df_1d, H3)
    L3_6h = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(H3_6h[i]) or np.isnan(L3_6h[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: require volume above average
        vol_filter = volume[i] > vol_ma[i]
        
        if position == 1:  # long position
            # Exit: price breaks below L3 or reaches H3 (take profit)
            if close[i] < L3_6h[i] or close[i] > H3_6h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above H3 or reaches L3 (take profit)
            if close[i] > H3_6h[i] or close[i] < L3_6h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for reversal entries with volume filter
            if vol_filter:
                # Long entry: price closes above L3 after touching or going below it
                if close[i] > L3_6h[i] and low[i] <= L3_6h[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short entry: price closes below H3 after touching or going above it
                elif close[i] < H3_6h[i] and high[i] >= H3_6h[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals