#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dCloud_Filter
Hypothesis: Use 1d Ichimoku cloud (Senkou Span A/B) as trend filter with 6h Tenkan-Kijun cross for entries.
In bull markets, price above cloud + TK cross up = long. In bear markets, price below cloud + TK cross down = short.
Volume confirmation reduces false signals. Designed for low trade frequency (target: 12-37/year) and works in both regimes.
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
    volume = prices['volume'].values
    
    # 1d data for Ichimoku cloud (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Ichimoku components: Tenkan (9), Kijun (26), Senkou Span A/B (52 displacement)
    # Tenkan-sen: (9-period high + 9-period low) / 2
    period9_high = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen: (26-period high + 26-period low) / 2
    period26_high = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A: (Tenkan + Kijun) / 2, plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B: (52-period high + 52-period low) / 2, plotted 26 periods ahead
    period52_high = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # The actual cloud is Senkou A/B shifted forward 26 periods
    # For filtering, we use current Senkou A/B values (which represent cloud 26 periods ago)
    # So we compare price to Senkou A/B values that are already shifted in the data
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # 6h Tenkan-Kijun for entries (faster, same logic)
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (period9_high_6h + period9_low_6h) / 2
    
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (period26_high_6h + period26_low_6h) / 2
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for 1d Ichimoku (52) and 6d indicators
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Cloud boundaries (top = max(A,B), bottom = min(A,B))
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Trend filter: price relative to cloud
        price_above_cloud = curr_close > cloud_top
        price_below_cloud = curr_close < cloud_bottom
        
        # TK cross: Tenkan crossing Kijun
        tk_cross_up = (tenkan_6h[i] > kijun_6h[i]) and (tenkan_6h[i-1] <= kijun_6h[i-1])
        tk_cross_down = (tenkan_6h[i] < kijun_6h[i]) and (tenkan_6h[i-1] >= kijun_6h[i-1])
        
        if position == 0:
            # Look for entry signals with volume spike
            # Long: price above cloud + TK cross up + volume spike
            long_entry = price_above_cloud and tk_cross_up and volume_spike[i]
            # Short: price below cloud + TK cross down + volume spike
            short_entry = price_below_cloud and tk_cross_down and volume_spike[i]
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price breaks below cloud or TK cross down
            if curr_close < cloud_bottom or tk_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price breaks above cloud or TK cross up
            if curr_close > cloud_top or tk_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dCloud_Filter"
timeframe = "6h"
leverage = 1.0