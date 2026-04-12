#!/usr/bin/env python3
"""
6h_1d_Ichimoku_Cloud_Breakout_v1
Hypothesis: Use Ichimoku Cloud from 1d as long-term trend filter and support/resistance, 
with TK (Tenkan-Kijun) cross on 6h for entry timing. 
Long when price is above cloud, TK crosses above KJ, and volume > 1.5x average.
Short when price is below cloud, TK crosses below KJ, and volume > 1.5x average.
Ichimoku provides robust trend definition; TK cross captures momentum; volume confirms strength.
Designed for low trade frequency (target: 50-150 total over 4 years) to minimize fee drag.
Works in bull via entries above cloud, in bear via entries below cloud.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Ichimoku_Cloud_Breakout_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: tenkan, kijun, senkou_a, senkou_b, chikou"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou = pd.Series(close).shift(26)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values, chikou.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Senkou B
        return np.zeros(n)
    
    # Calculate Ichimoku on daily data
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values
    )
    
    # Align Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    chikou_1d_aligned = align_htf_to_ltf(prices, df_1d, chikou_1d)
    
    # Cloud top and bottom (Senkou A and B)
    cloud_top = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    cloud_bottom = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    
    # TK cross signals on 6h
    # Calculate TK on 6h price data
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan_6h = (period9_high_6h + period9_low_6h) / 2
    
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun_6h = (period26_high_6h + period26_low_6h) / 2
    
    tk_cross_above = (tenkan_6h > kijun_6h) & (tenkan_6h.shift(1) <= kijun_6h.shift(1))
    tk_cross_below = (tenkan_6h < kijun_6h) & (tenkan_6h.shift(1) >= kijun_6h.shift(1))
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = volume_series / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values  # default to 1.0 if no MA
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after Ichimoku warmup
        # Skip if any data invalid
        if (np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(tk_cross_above[i]) if i < len(tk_cross_above) else False or
            np.isnan(tk_cross_below[i]) if i < len(tk_cross_below) else False or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Ensure we have valid indices for TK cross arrays
        if i >= len(tk_cross_above) or i >= len(tk_cross_below):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        # Long conditions: price above cloud, TK crosses above KJ, volume confirmation
        price_above_cloud = close[i] > cloud_top[i]
        long_signal = price_above_cloud and tk_cross_above[i] and (vol_ratio[i] > 1.5)
        
        # Short conditions: price below cloud, TK crosses below KJ, volume confirmation
        price_below_cloud = close[i] < cloud_bottom[i]
        short_signal = price_below_cloud and tk_cross_below[i] and (vol_ratio[i] > 1.5)
        
        # Exit conditions: TK cross in opposite direction
        long_exit = tk_cross_below[i]
        short_exit = tk_cross_above[i]
        
        # Signal logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals