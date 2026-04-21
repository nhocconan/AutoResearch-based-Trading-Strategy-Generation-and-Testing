#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_v2
Hypothesis: Use Ichimoku cloud from daily timeframe as trend filter (bullish when price above cloud, bearish when below) and 6h Tenkan-Kijun cross for entry timing. 
This combines higher timeframe trend structure with lower timeframe momentum signals, reducing whipsaws in sideways markets.
Works in bull markets by buying dips to Tenkan in uptrends (price > cloud) and in bear markets by selling rallies to Tenkan in downtrends (price < cloud).
Volume confirmation ensures institutional participation.
Target: 12-37 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need enough for Ichimoku calculations
        return np.zeros(n)
    
    # === Daily Ichimoku components ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Cloud top and bottom (Senkou Span A and B)
    cloud_top = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # === 6h Tenkan-Kijun cross for entry timing ===
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    
    period9_high_6h = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan_6h_internal = (period9_high_6h + period9_low_6h) / 2
    
    period26_high_6h = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun_6h_internal = (period26_high_6h + period26_low_6h) / 2
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after warmup for Ichimoku
        # Skip if indicators not ready
        if (np.isnan(tenkan_6h[i]) or 
            np.isnan(kijun_6h[i]) or 
            np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i]) or 
            np.isnan(tenkan_6h_internal[i]) or 
            np.isnan(kijun_6h_internal[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_open = prices['open'].iloc[i]
        
        # Bullish conditions: price above cloud + Tenkan crosses above Kijun + volume
        bullish_cross = tenkan_6h_internal[i] > kijun_6h_internal[i] and tenkan_6h_internal[i-1] <= kijun_6h_internal[i-1]
        price_above_cloud = price_close > cloud_top[i]
        
        # Bearish conditions: price below cloud + Tenkan crosses below Kijun + volume
        bearish_cross = tenkan_6h_internal[i] < kijun_6h_internal[i] and tenkan_6h_internal[i-1] >= kijun_6h_internal[i-1]
        price_below_cloud = price_close < cloud_bottom[i]
        
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Long: price above cloud + bullish TK cross + volume
            if (price_above_cloud and
                bullish_cross and
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price below cloud + bearish TK cross + volume
            elif (price_below_cloud and
                  bearish_cross and
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when Tenkan-Kijun cross reverses
            if position == 1 and (tenkan_6h_internal[i] < kijun_6h_internal[i]):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (tenkan_6h_internal[i] > kijun_6h_internal[i]):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_v2"
timeframe = "6h"
leverage = 1.0