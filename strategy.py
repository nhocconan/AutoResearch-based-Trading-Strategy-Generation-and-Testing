#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_1dTrend
Hypothesis: Ichimoku TK cross on 6h with 1d cloud filter. In bull markets (price > 1d cloud), long on TK cross up; in bear markets (price < 1d cloud), short on TK cross down. Cloud acts as dynamic support/resistance and regime filter. Designed for 6h timeframe targeting 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits. Uses discrete position sizing (0.25) to minimize fee churn. Works in bull/bear markets: In bull regime (price above cloud), TK cross up captures momentum continuation; in bear regime (price below cloud), TK cross down captures continuation. Exit when price re-enters cloud (regime change or consolidation).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for Ichimoku cloud (Senkou Span A/B) and price position
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = df_1d['high'].rolling(window=9, min_periods=9).max().values
    period9_low = df_1d['low'].rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = df_1d['high'].rolling(window=26, min_periods=26).max().values
    period26_low = df_1d['low'].rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = df_1d['high'].rolling(window=52, min_periods=52).max().values
    period52_low = df_1d['low'].rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Shift Senkou Spans forward by 26 periods (cloud is plotted ahead)
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    # Fill first 26 values with NaN (no data)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Align Ichimoku components to 6h
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_shifted)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_shifted)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25  # Discrete size to reduce fee churn
    
    # Warmup: need Ichimoku calculations (max 52 + 26 shift)
    start_idx = 78  # 52 + 26
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        
        # Cloud boundaries (Senkou Span A and B form the cloud)
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        if position == 0:
            # Look for entry: TK cross with cloud as regime filter
            # Bullish TK cross: Tenkan crosses above Kijun
            tk_cross_up = (tenkan_val > kijun_val) and (tenkan_aligned[i-1] <= kijun_aligned[i-1])
            # Bearish TK cross: Tenkan crosses below Kijun
            tk_cross_down = (tenkan_val < kijun_val) and (tenkan_aligned[i-1] >= kijun_aligned[i-1])
            
            # Long condition: price above cloud (bullish regime) + bullish TK cross
            long_condition = (close_val > cloud_top) and tk_cross_up
            # Short condition: price below cloud (bearish regime) + bearish TK cross
            short_condition = (close_val < cloud_bottom) and tk_cross_down
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price re-enters cloud (regime change or consolidation)
            if close_val < cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price re-enters cloud (regime change or consolidation)
            if close_val > cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1dTrend"
timeframe = "6h"
leverage = 1.0