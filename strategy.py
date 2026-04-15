#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla Pivot Reversion with Volume Spike
# Uses 1d Camarilla pivot levels (H3, L3) for mean reversion entries.
# Long when price touches L3 with volume spike, short when touches H3 with volume spike.
# Exits at opposite H3/L3 or when volume dries up.
# Works in ranging markets (reversion) and trending markets (breakouts at H4/L4).
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # H4 = Pivot + Range * 1.1/2
    # H3 = Pivot + Range * 1.1/4
    # L3 = Pivot - Range * 1.1/4
    # L4 = Pivot - Range * 1.1/2
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_1d = df_1d['high'] - df_1d['low']
    
    pivot = typical_price.values
    H3 = pivot + range_1d.values * 1.1 / 4
    L3 = pivot - range_1d.values * 1.1 / 4
    H4 = pivot + range_1d.values * 1.1 / 2
    L4 = pivot - range_1d.values * 1.1 / 2
    
    # Shift to avoid look-ahead (use previous day's levels)
    H3 = np.roll(H3, 1)
    L3 = np.roll(L3, 1)
    H4 = np.roll(H4, 1)
    L4 = np.roll(L4, 1)
    H3[0] = np.nan
    L3[0] = np.nan
    H4[0] = np.nan
    L4[0] = np.nan
    
    # Align to 6h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Volume spike detection (volume > 2x 20-period median)
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(vol_median[i])):
            continue
        
        # Long entry: price touches L3 with volume spike, not already long
        if (low[i] <= L3_aligned[i] and
            volume[i] > 2.0 * vol_median[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price touches H3 with volume spike, not already short
        elif (high[i] >= H3_aligned[i] and
              volume[i] > 2.0 * vol_median[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit conditions
        elif position == 1:
            # Exit long: price reaches H3 (opposite level) or volume dries up
            if (high[i] >= H3_aligned[i] or
                volume[i] < 0.5 * vol_median[i]):
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit short: price reaches L3 (opposite level) or volume dries up
            if (low[i] <= L3_aligned[i] or
                volume[i] < 0.5 * vol_median[i]):
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_Pivot_Reversion_Volume"
timeframe = "6h"
leverage = 1.0