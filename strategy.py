#!/usr/bin/env python3
"""
6h_ichimoku_12h_trend_volume_v1
Hypothesis: On 6-hour timeframe, use Ichimoku cloud from 12-hour timeframe for trend direction, with Tenkan/Kijun cross for entry timing and volume confirmation. The Ichimoku cloud provides strong support/resistance levels and future cloud acts as leading indicator. Works in bull markets (buy when price above cloud in uptrend) and bear markets (sell when price below cloud in downtrend) by using higher timeframe trend filter. Volume confirmation ensures breakouts have conviction. Targets 12-37 trades/year to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_12h_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Ichimoku
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 12h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    tenkan_sen = []
    for i in range(len(high_12h)):
        if i < 8:
            tenkan_sen.append(np.nan)
        else:
            tenkan_sen.append((np.max(high_12h[i-8:i+1]) + np.min(low_12h[i-8:i+1])) / 2)
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = []
    for i in range(len(high_12h)):
        if i < 25:
            kijun_sen.append(np.nan)
        else:
            kijun_sen.append((np.max(high_12h[i-25:i+1]) + np.min(low_12h[i-25:i+1])) / 2)
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = []
    for i in range(len(tenkan_sen)):
        if i < 25 or np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]):
            senkou_span_a.append(np.nan)
        else:
            senkou_span_a.append((tenkan_sen[i] + kijun_sen[i]) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = []
    for i in range(len(high_12h)):
        if i < 51:
            senkou_span_b.append(np.nan)
        else:
            senkou_span_b.append((np.max(high_12h[i-51:i+1]) + np.min(low_12h[i-51:i+1])) / 2)
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind
    chikou_span = df_12h['close'].values
    
    # Align Ichimoku components to 6h timeframe (shifted by 1 for completed bars only)
    tenkan_aligned = align_htf_to_ltf(prices, df_12h, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_12h, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_b)
    chikou_aligned = align_htf_to_ltf(prices, df_12h, chikou_span)
    
    # Calculate 50-period average volume for confirmation on 6h
    vol_avg = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if Ichimoku data not available
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend: price above/below cloud
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Tenkan/Kijun cross for entry timing
        tenkan_now = tenkan_aligned[i]
        kijun_now = kijun_aligned[i]
        tenkan_prev = tenkan_aligned[i-1]
        kijun_prev = kijun_aligned[i-1]
        
        tk_cross_up = (tenkan_prev <= kijun_prev) and (tenkan_now > kijun_now)
        tk_cross_down = (tenkan_prev >= kijun_prev) and (tenkan_now < kijun_now)
        
        # Volume confirmation: current volume > 1.5x 50-period average
        vol_confirm = volume[i] > 1.5 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        if position == 1:  # Long position
            # Exit when price crosses below cloud or Tenkan/Kijun cross down
            if price_below_cloud or tk_cross_down:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price crosses above cloud or Tenkan/Kijun cross up
            if price_above_cloud or tk_cross_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price above cloud + Tenkan crosses above Kijun + volume confirmation
            long_entry = price_above_cloud and tk_cross_up and vol_confirm
            # Short entry: price below cloud + Tenkan crosses below Kijun + volume confirmation
            short_entry = price_below_cloud and tk_cross_down and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals