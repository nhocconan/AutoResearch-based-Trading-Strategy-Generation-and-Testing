#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout
Hypothesis: 6h Ichimoku cloud (from 1d data) breakout with Tenkan/Kijun cross and volume confirmation.
- Tenkan-sen (9-period) crossing above/below Kijun-sen (26-period) signals momentum shift.
- Price breaking above/below cloud (Senkou Span A/B) confirms trend strength.
- Cloud acts as dynamic support/resistance; breakouts with volume indicate institutional interest.
- Works in bull/bear via cloud filter and momentum confirmation.
Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.
"""

name = "6h_Ichimoku_Cloud_Breakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku calculation (slow but stable)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 days for Senkou Span B
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Cloud boundaries
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Tenkan/Kijun cross
        tk_cross_above = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_cross_below = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        if position == 0:
            # Long: price breaks above cloud + TK cross bullish + volume
            if (close[i] > upper_cloud and 
                tk_cross_above and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud + TK cross bearish + volume
            elif (close[i] < lower_cloud and 
                  tk_cross_below and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below cloud or TK cross turns bearish
            if (close[i] < lower_cloud) or (not tk_cross_above):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above cloud or TK cross turns bullish
            if (close[i] > upper_cloud) or (tk_cross_above):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals