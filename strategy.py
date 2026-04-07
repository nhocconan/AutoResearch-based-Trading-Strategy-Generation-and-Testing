#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_volume_v1
Hypothesis: On 12-hour timeframe, use Camarilla pivot levels from daily timeframe for mean reversion entries with volume confirmation.
Long when price touches Camarilla L3 level with volume > 1.5x 20-period average.
Short when price touches Camarilla H3 level with volume > 1.5x 20-period average.
Exit when price reaches Camarilla L4/H4 levels or returns to pivot point.
Designed for 12-37 trades/year to minimize fee drag while capturing mean reversion in ranging markets.
Works in both bull/bear markets as Camarilla levels adapt to volatility and volume filter avoids false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    L4 = close_1d - range_1d * 1.1 / 2
    L3 = close_1d - range_1d * 1.1 / 4
    L2 = close_1d - range_1d * 1.1 / 6
    L1 = close_1d - range_1d * 1.1 / 12
    H1 = close_1d + range_1d * 1.1 / 12
    H2 = close_1d + range_1d * 1.1 / 6
    H3 = close_1d + range_1d * 1.1 / 4
    H4 = close_1d + range_1d * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    
    # Volume filter: 20-period average on 12h timeframe
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(H3_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(H4_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches L4 (target) or returns to pivot (reversal)
            if close[i] <= L4_aligned[i] or close[i] >= pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches H4 (target) or returns to pivot (reversal)
            if close[i] >= H4_aligned[i] or close[i] <= pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation
            if vol_ok:
                # Long: price touches L3 level (support)
                if low[i] <= L3_aligned[i] and close[i] > L3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price touches H3 level (resistance)
                elif high[i] >= H3_aligned[i] and close[i] < H3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals