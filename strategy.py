#!/usr/bin/env python3
name = "6h_Ichimoku_Cloud_Filter_1dTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # Get daily data for Ichimoku and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    tenkan_sen = np.full(len(df_1d), np.nan)
    for i in range(8, len(df_1d)):
        tenkan_sen[i] = (np.max(high_1d[i-8:i+1]) + np.min(low_1d[i-8:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    kijun_sen = np.full(len(df_1d), np.nan)
    for i in range(25, len(df_1d)):
        kijun_sen[i] = (np.max(high_1d[i-25:i+1]) + np.min(low_1d[i-25:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    senkou_span_b = np.full(len(df_1d), np.nan)
    for i in range(51, len(df_1d)):
        senkou_span_b[i] = (np.max(high_1d[i-51:i+1]) + np.min(low_1d[i-51:i+1])) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # 1d trend filter: price above/below Kumo (cloud)
    # Cloud top = max(Senkou Span A, Senkou Span B)
    # Cloud bottom = min(Senkou Span A, Senkou Span B)
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # TK Cross signals
    tk_cross_up = tenkan_sen_aligned > kijun_sen_aligned
    tk_cross_down = tenkan_sen_aligned < kijun_sen_aligned
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 6  # ~1.5 days for 6h to reduce trades
    
    start_idx = max(100, 20, 52)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: TK cross up + price above cloud + volume filter
            if (tk_cross_up[i] and 
                price_above_cloud[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: TK cross down + price below cloud + volume filter
            elif (tk_cross_down[i] and 
                  price_below_cloud[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: TK cross down or price enters cloud
            if (tk_cross_down[i] or 
                (close[i] <= cloud_top[i] and close[i] >= cloud_bottom[i])):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TK cross up or price enters cloud
            if (tk_cross_up[i] or 
                (close[i] <= cloud_top[i] and close[i] >= cloud_bottom[i])):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Ichimoku cloud provides dynamic support/resistance that adapts to volatility,
# while TK cross signals momentum shifts. Using 1d Ichimoku on 6h timeframe filters
# false signals and captures multi-day trends. Works in bull markets (trend following
# with cloud support) and bear markets (trend following with cloud resistance).
# Volume confirmation and cooldown reduce trades to target 50-150 over 4 years.