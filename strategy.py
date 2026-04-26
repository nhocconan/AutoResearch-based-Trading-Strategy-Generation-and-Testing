#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_v1
Hypothesis: 6h Ichimoku Tenkan-Kijun cross with 1d cloud filter (price above/below cloud) and volume confirmation (1.5x 20-period average). 
Ichimoku provides dynamic support/resistance via cloud, TK cross signals momentum shift, and 1d cloud filter ensures alignment with higher timeframe trend. 
Volume spike filters low-conviction breakouts. Designed for 6h timeframe to capture medium-term swings in both bull and bear markets.
Target: 50-150 trades over 4 years (12-37/year) with discrete sizing (0.25) to minimize fee drag.
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
    
    # Load 1d data ONCE before loop for Ichimoku components and cloud filter
    df_1d = get_htf_data(prices, '1d')
    
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
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind (not used for signals)
    
    # Calculate 1d Ichimoku cloud for trend filter
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # 1d Tenkan-sen
    period9_high_1d = pd.Series(df_1d_high).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(df_1d_low).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high_1d + period9_low_1d) / 2
    
    # 1d Kijun-sen
    period26_high_1d = pd.Series(df_1d_high).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(df_1d_low).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high_1d + period26_low_1d) / 2
    
    # 1d Senkou Span A
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # 1d Senkou Span B
    period52_high_1d = pd.Series(df_1d_high).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(df_1d_low).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = ((period52_high_1d + period52_low_1d) / 2)
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Volume filter: volume > 1.5 * volume_ma(20)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B, 26 for TK cross alignment, 20 for volume MA)
    start_idx = max(52, 26, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        top_cloud = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        bottom_cloud = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Determine 1d cloud boundaries for trend filter
        top_cloud_1d = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        bottom_cloud_1d = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # Ichimoku signals:
        # TK cross: Tenkan crosses above/below Kijun
        tk_cross_up = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
        tk_cross_down = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
        
        # Price position relative to cloud
        price_above_cloud = close[i] > top_cloud
        price_below_cloud = close[i] < bottom_cloud
        
        # 1d trend filter: price relative to 1d cloud
        trend_up = close[i] > top_cloud_1d  # Bullish if price above 1d cloud
        trend_down = close[i] < bottom_cloud_1d  # Bearish if price below 1d cloud
        
        if position == 0:
            # Long: TK cross up AND price above cloud AND 1d bullish trend AND volume spike
            if tk_cross_up and price_above_cloud and trend_up and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: TK cross down AND price below cloud AND 1d bearish trend AND volume spike
            elif tk_cross_down and price_below_cloud and trend_down and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: TK cross down OR price falls below cloud OR 1d trend turns bearish
            if tk_cross_down or close[i] < bottom_cloud or not trend_up:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TK cross up OR price rises above cloud OR 1d trend turns bullish
            if tk_cross_up or close[i] > top_cloud or not trend_down:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_v1"
timeframe = "6h"
leverage = 1.0