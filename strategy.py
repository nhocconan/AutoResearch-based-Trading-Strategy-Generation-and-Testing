#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_Filter_v1
Hypothesis: Ichimoku TK cross (Tenkan/Kijun) with 1d cloud filter on 6h timeframe. 
Long when TK crosses above AND price > 1d cloud (Senkou Span A/B max). 
Short when TK crosses below AND price < 1d cloud (Senkou Span A/B min).
Uses volume confirmation (1.5x 20-period MA) to reduce false breaks.
Designed for medium frequency (50-150/year) with discrete sizing (0.25) to balance capture and fees.
Works in both bull/bear via cloud as dynamic support/resistance and trend filter.
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
    
    # Get 1d data for Ichimoku cloud (Senkou Span A/B)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need 26*2 for Senkou Span B
        return np.zeros(n)
    
    # Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Cloud boundaries (future cloud, but we use aligned values so it's lagged correctly)
    senkou_span_a = senkou_a
    senkou_span_b = senkou_b
    
    # The cloud: between Senkou Span A and B
    # Actual cloud top/bottom for filtering (we want price relative to cloud)
    cloud_top = np.maximum(senkou_span_a, senkou_span_b)
    cloud_bottom = np.minimum(senkou_span_a, senkou_span_b)
    
    # Align 1d Ichimoku to 6h (aligned gives completed cloud values)
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    cloud_top_6h = align_htf_to_ltf(prices, df_1d, cloud_top)
    cloud_bottom_6h = align_htf_to_ltf(prices, df_1d, cloud_bottom)
    
    # TK cross signals: Tenkan crosses Kijun
    # Bullish cross: Tenkan > Kijun and previous Tenkan <= previous Kijun
    # Bearish cross: Tenkan < Kijun and previous Tenkan >= previous Kijun
    tk_bullish = (tenkan_6h > kijun_6h) & (np.roll(tenkan_6h, 1) <= np.roll(kijun_6h, 1))
    tk_bearish = (tenkan_6h < kijun_6h) & (np.roll(tenkan_6h, 1) >= np.roll(kijun_6h, 1))
    # Handle first element
    tk_bullish[0] = False
    tk_bearish[0] = False
    
    # Volume confirmation: 1.5x average volume (moderate to avoid overtrading)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku calculations (52) and volume MA (20)
    start_idx = max(52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_6h[i]) or 
            np.isnan(kijun_6h[i]) or 
            np.isnan(cloud_top_6h[i]) or 
            np.isnan(cloud_bottom_6h[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        tenkan_val = tenkan_6h[i]
        kijun_val = kijun_6h[i]
        cloud_top_val = cloud_top_6h[i]
        cloud_bottom_val = cloud_bottom_6h[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: TK bullish cross AND price above cloud AND volume confirmation
            long_signal = tk_bullish[i] and (close_val > cloud_top_val) and (volume_val > 1.5 * vol_ma_val)
            # Short: TK bearish cross AND price below cloud AND volume confirmation
            short_signal = tk_bearish[i] and (close_val < cloud_bottom_val) and (volume_val > 1.5 * vol_ma_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: TK bearish cross OR price drops below cloud bottom
            if tk_bearish[i] or (close_val < cloud_bottom_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TK bullish cross OR price rises above cloud top
            if tk_bullish[i] or (close_val > cloud_top_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_Filter_v1"
timeframe = "6h"
leverage = 1.0