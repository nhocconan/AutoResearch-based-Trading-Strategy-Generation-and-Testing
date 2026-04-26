#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_v1
Hypothesis: Trade Ichimoku TK cross with 1d cloud filter on 6h timeframe. 
Long when Tenkan > Kijun and price > 1d Senkou Span A/B (bullish cloud). 
Short when Tenkan < Kijun and price < 1d Senkou Span A/B (bearish cloud).
Uses volume confirmation (1.5x median) to reduce false signals.
Designed to capture trends while avoiding chop via cloud filter. 
Target: 50-150 total trades over 4 years (12-37/year) with discrete position sizing 0.25.
Works in bull markets via trend continuation and bear markets via short signals during downtrends.
"""

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
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Get 1d data for HTF cloud filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate 1d Ichimoku cloud (Senkou Span A/B)
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    
    # 1d Tenkan-sen (9-period)
    period9_high_1d = pd.Series(df_1d_high).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(df_1d_low).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high_1d + period9_low_1d) / 2
    
    # 1d Kijun-sen (26-period)
    period26_high_1d = pd.Series(df_1d_high).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(df_1d_low).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high_1d + period26_low_1d) / 2
    
    # 1d Senkou Span A and B
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    period52_high_1d = pd.Series(df_1d_high).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(df_1d_low).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = ((period52_high_1d + period52_low_1d) / 2)
    
    # Align 1d cloud to 6h timeframe (cloud values are plotted 26 periods ahead)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Volume confirmation: 1.5x median volume (reduces churn)
    vol_median = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku calculations (52) and volume median (30)
    start_idx = max(52, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(vol_median[i])):
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            continue
        
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        close_val = close[i]
        senkou_a_val = senkou_a[i]
        senkou_b_val = senkou_b[i]
        senkou_a_1d_val = senkou_a_1d_aligned[i]
        senkou_b_1d_val = senkou_b_1d_aligned[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        
        # Determine cloud boundaries (top and bottom of cloud)
        cloud_top = max(senkou_a_1d_val, senkou_b_1d_val)
        cloud_bottom = min(senkou_a_1d_val, senkou_b_1d_val)
        
        if position == 0:
            # Long: TK cross bullish (Tenkan > Kijun) AND price above 1d cloud AND volume confirmation
            long_signal = (tenkan_val > kijun_val) and \
                          (close_val > cloud_top) and \
                          (volume_val > 1.5 * vol_median_val)
            # Short: TK cross bearish (Tenkan < Kijun) AND price below 1d cloud AND volume confirmation
            short_signal = (tenkan_val < kijun_val) and \
                           (close_val < cloud_bottom) and \
                           (volume_val > 1.5 * vol_median_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long: exit when TK cross turns bearish OR price drops below cloud bottom
            signals[i] = 0.25
            exit_signal = (tenkan_val < kijun_val) or (close_val < cloud_bottom)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short: exit when TK cross turns bullish OR price rises above cloud top
            signals[i] = -0.25
            exit_signal = (tenkan_val > kijun_val) or (close_val > cloud_top)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_v1"
timeframe = "6h"
leverage = 1.0