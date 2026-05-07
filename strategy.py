#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_With_1dTrend
Hypothesis: Use Ichimoku cloud from daily timeframe for trend bias and 6h Tenkan/Kijun cross for entry timing. Enter long when price breaks above cloud in daily uptrend, short when breaks below cloud in daily downtrend. Uses volume confirmation to avoid false breaks. Designed to work in both bull and bear markets by following daily trend while using Ichimoku for objective support/resistance. Target: 15-30 trades/year to minimize fee drag.
"""

name = "6h_Ichimoku_Cloud_Breakout_With_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan, Kijun, Senkou A/B, Chikou"""
    n1 = 9
    n2 = 26
    n3 = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan = (pd.Series(high).rolling(window=n1, min_periods=n1).max() + 
              pd.Series(low).rolling(window=n1, min_periods=n1).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun = (pd.Series(high).rolling(window=n2, min_periods=n2).max() + 
             pd.Series(low).rolling(window=n2, min_periods=n2).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high).rolling(window=n3, min_periods=n3).max() + 
                 pd.Series(low).rolling(window=n3, min_periods=n3).min()) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou = pd.Series(close)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values, chikou.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate Ichimoku on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align daily Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    chikou_1d_aligned = align_htf_to_ltf(prices, df_1d, chikou_1d)
    
    # Daily trend determination: price above/below cloud + Chikou confirmation
    # Cloud top/bottom
    cloud_top = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    cloud_bottom = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    
    # Price relative to cloud
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # Daily close aligned for Chikou comparison
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Chikou above/below price (26 periods ago)
    chikou_above_price = chikou_1d_aligned > daily_close_aligned
    chikou_below_price = chikou_1d_aligned < daily_close_aligned
    
    # Daily trend: bullish if price above cloud AND Chikou above price
    daily_trend_up = price_above_cloud & chikou_above_price
    # Daily trend: bearish if price below cloud AND Chikou below price
    daily_trend_down = price_below_cloud & chikou_below_price
    
    # Calculate Ichimoku on 6h for entry signals
    tenkan_6h, kijun_6h, _, _, _ = calculate_ichimoku(high, low, close)
    
    # TK cross signals
    tk_cross_up = (tenkan_6h > kijun_6h) & (np.roll(tenkan_6h, 1) <= np.roll(kijun_6h, 1))
    tk_cross_down = (tenkan_6h < kijun_6h) & (np.roll(tenkan_6h, 1) >= np.roll(kijun_6h, 1))
    
    # Volume confirmation: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Warmup for Ichimoku calculations
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(daily_close_aligned[i]) or np.isnan(chikou_1d_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TK cross up in daily uptrend with volume confirmation
            if (tk_cross_up[i] and 
                daily_trend_up[i] and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: TK cross down in daily downtrend with volume confirmation
            elif (tk_cross_down[i] and 
                  daily_trend_down[i] and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TK cross down or trend turns down
            if (tk_cross_down[i] or not daily_trend_up[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TK cross up or trend turns up
            if (tk_cross_up[i] or not daily_trend_down[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals