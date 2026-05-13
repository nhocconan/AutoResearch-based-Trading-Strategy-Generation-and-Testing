# 165128
#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Reversal_With_Volume_Filter
Hypothesis: Weekly pivot levels act as strong support/resistance in crypto markets. A rejection (price touches but fails to break) of weekly R2/S2 with volume confirmation signals mean reversion. Works in both bull/bear markets as pivots adapt to volatility. Uses weekly timeframe for structure and 6h for entry timing to achieve low trade frequency (~25-40/year) minimizing fee drag.
"""

name = "6h_Weekly_Pivot_Reversal_With_Volume_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation (once before loop)
    df_w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points from previous week
    # Pivot Point = (High + Low + Close) / 3
    pp = (df_w['high'] + df_w['low'] + df_w['close']) / 3
    # Weekly range
    range_w = df_w['high'] - df_w['low']
    
    # Weekly support/resistance levels
    # R2 = PP + (2 * (High - Low))
    # S2 = PP - (2 * (High - Low))
    weekly_r2 = pp + (2 * range_w)
    weekly_s2 = pp - (2 * range_w)
    
    # Align to 6h - use previous week's levels (available at 6h open)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_w, weekly_r2.values)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_w, weekly_s2.values)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Price touches/rejects S2 with volume confirmation
            if (low[i] <= weekly_s2_aligned[i] * 1.001 and  # Allow small tolerance
                close[i] > weekly_s2_aligned[i] and         # Close back above S2
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches/rejects R2 with volume confirmation
            elif (high[i] >= weekly_r2_aligned[i] * 0.999 and  # Allow small tolerance
                  close[i] < weekly_r2_aligned[i] and         # Close back below R2
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches midpoint or shows rejection at R1
            r1 = weekly_s2_aligned[i] + (weekly_r2_aligned[i] - weekly_s2_aligned[i]) / 3  # Approximate R1
            if (close[i] >= r1 or 
                not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches midpoint or shows rejection at S1
            s1 = weekly_s2_aligned[i] + 2 * (weekly_r2_aligned[i] - weekly_s2_aligned[i]) / 3  # Approximate S1
            if (close[i] <= s1 or 
                not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals