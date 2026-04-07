#!/usr/bin/env python3
"""
6h_ichimoku_cloud_1d_trend_volume_v1
Hypothesis: Use Ichimoku cloud from 1d timeframe for trend filtering, with TK cross on 6h for entry signals and volume confirmation. The cloud acts as dynamic support/resistance, providing high-probability entries in both bull and bear markets. Long when price > cloud, TK crosses bullish, and volume confirms; short when price < cloud, TK crosses bearish, and volume confirms. This reduces false signals and captures strong trends with proper filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
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
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_span_a_6h, senkou_span_b_6h)
    cloud_bottom = np.minimum(senkou_span_a_6h, senkou_span_b_6h)
    
    # TK Cross signals on 6h
    # Tenkan-sen crossing above Kijun-sen (bullish)
    tk_bullish = (tenkan_sen_6h > kijun_sen_6h) & (tenkan_sen_6h <= kijun_sen_6h)
    # Tenkan-sen crossing below Kijun-sen (bearish)
    tk_bearish = (tenkan_sen_6h < kijun_sen_6h) & (tenkan_sen_6h >= kijun_sen_6h)
    # Fix the crossover detection (proper edge detection)
    tk_bullish = (tenkan_sen_6h > kijun_sen_6h) & (np.roll(tenkan_sen_6h, 1) <= np.roll(kijun_sen_6h, 1))
    tk_bearish = (tenkan_sen_6h < kijun_sen_6h) & (np.roll(tenkan_sen_6h, 1) >= np.roll(kijun_sen_6h, 1))
    # Handle first element
    tk_bullish[0] = False
    tk_bearish[0] = False
    
    # Volume confirmation (20-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(26, n):  # Start after Kijun period
        # Skip if required data not available
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Price position relative to cloud
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price drops below cloud
            if price_below_cloud:
                exit_long = True
            # Exit if TK turns bearish
            elif tk_bearish[i]:
                exit_long = True
            # Exit if TK cross is bearish (Tenkan < Kijun)
            elif tenkan_sen_6h[i] < kijun_sen_6h[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price rises above cloud
            if price_above_cloud:
                exit_short = True
            # Exit if TK turns bullish
            elif tk_bullish[i]:
                exit_short = True
            # Exit if TK cross is bullish (Tenkan > Kijun)
            elif tenkan_sen_6h[i] > kijun_sen_6h[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry conditions
            long_entry = False
            # Price above cloud, TK bullish cross, and volume confirmation
            if price_above_cloud and tk_bullish[i] and vol_confirm:
                long_entry = True
            
            # Short entry conditions
            short_entry = False
            # Price below cloud, TK bearish cross, and volume confirmation
            if price_below_cloud and tk_bearish[i] and vol_confirm:
                short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals