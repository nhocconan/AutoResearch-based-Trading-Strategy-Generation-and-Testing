#!/usr/bin/env python3
# 6h_1d_1w_ichimoku_cloud_trend_v1
# Strategy: Ichimoku Cloud trend following on 6h with daily cloud filter
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Ichimoku Cloud provides strong trend identification. Use daily cloud color (Senkou Span A/B) as higher timeframe trend filter. Enter long when 6h price is above both daily cloud and 6d Tenkan-Kijun cross, short when below. This avoids whipsaws by aligning with higher timeframe trend. Works in both bull and bear markets by following the dominant trend on daily timeframe.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_ichimoku_cloud_trend_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan, Kijun, Senkou Span A/B, Chikou"""
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop for cloud filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate Ichimoku on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align daily Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Calculate Ichimoku on 6h data for entry signals
    tenkan_6h, kijun_6h, _, _ = calculate_ichimoku(high, low, close)
    
    # Cloud top and bottom (Senkou Span A/B)
    daily_cloud_top = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    daily_cloud_bottom = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if any required data is invalid
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(daily_cloud_top[i]) or np.isnan(daily_cloud_bottom[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine daily trend from cloud color
        # Green cloud (bullish): Senkou A > Senkou B
        # Red cloud (bearish): Senkou A < Senkou B
        bullish_cloud = senkou_a_1d_aligned[i] > senkou_b_1d_aligned[i]
        bearish_cloud = senkou_a_1d_aligned[i] < senkou_b_1d_aligned[i]
        
        # 6h Tenkan-Kijun cross
        tk_cross_bull = tenkan_6h[i] > kijun_6h[i]
        tk_cross_bear = tenkan_6h[i] < kijun_6h[i]
        
        # Entry logic: Ichimoku signals aligned with daily cloud
        if (close[i] > daily_cloud_top[i] and tk_cross_bull and bullish_cloud and position != 1):
            # Strong bullish: price above cloud, TK cross bull, daily cloud bullish
            position = 1
            signals[i] = 0.25
        elif (close[i] < daily_cloud_bottom[i] and tk_cross_bear and bearish_cloud and position != -1):
            # Strong bearish: price below cloud, TK cross bear, daily cloud bearish
            position = -1
            signals[i] = -0.25
        # Exit: TK cross reverses or price enters cloud
        elif position == 1 and (tk_cross_bear or close[i] < daily_cloud_top[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (tk_cross_bull or close[i] > daily_cloud_bottom[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals