#!/usr/bin/env python3
name = "6h_Weekly_Ichimoku_Cloud_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Ichimoku (higher timeframe for trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # Need enough data for weekly calculations
        return np.zeros(n)
    
    # Calculate Ichimoku components on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    tenkan_sen_1w = np.full_like(close_1w, np.nan)
    for i in range(len(close_1w)):
        if i >= period_tenkan - 1:
            tenkan_sen_1w[i] = (np.max(high_1w[i-period_tenkan+1:i+1]) + np.min(low_1w[i-period_tenkan+1:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    kijun_sen_1w = np.full_like(close_1w, np.nan)
    for i in range(len(close_1w)):
        if i >= period_kijun - 1:
            kijun_sen_1w[i] = (np.max(high_1w[i-period_kijun+1:i+1]) + np.min(low_1w[i-period_kijun+1:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
    senkou_span_a_1w = (tenkan_sen_1w + kijun_sen_1w) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2, plotted 26 periods ahead
    period_senkou_b = 52
    senkou_span_b_1w = np.full_like(close_1w, np.nan)
    for i in range(len(close_1w)):
        if i >= period_senkou_b - 1:
            senkou_span_b_1w[i] = (np.max(high_1w[i-period_senkou_b+1:i+1]) + np.min(low_1w[i-period_senkou_b+1:i+1])) / 2
    
    # Chikou Span (Lagging Span): Current close plotted 26 periods back
    # For signal generation, we use current price vs cloud from 26 periods ago
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen_1w)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen_1w)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a_1w)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b_1w)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Volume confirmation: 20-period average volume
    vol_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_20_1d)
    
    # 6-period EMA for entry timing on 6h chart
    ema6_6h = pd.Series(close).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 60  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(vol_20_1d_aligned[i]) or np.isnan(ema6_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Check if price is above/below cloud
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        
        # Volume confirmation: current volume > 20-period average
        volume_confirm = volume[i] > vol_20_1d_aligned[i]
        
        if position == 0:
            # Long: Price above cloud, Tenkan > Kijun (bullish TK cross), volume confirmation
            if (price_above_cloud and 
                tenkan_sen_aligned[i] > kijun_sen_aligned[i] and
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud, Tenkan < Kijun (bearish TK cross), volume confirmation
            elif (price_below_cloud and 
                  tenkan_sen_aligned[i] < kijun_sen_aligned[i] and
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price falls below cloud or TK cross turns bearish
            if (not price_above_cloud or 
                tenkan_sen_aligned[i] <= kijun_sen_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price rises above cloud or TK cross turns bullish
            if (not price_below_cloud or 
                tenkan_sen_aligned[i] >= kijun_sen_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals