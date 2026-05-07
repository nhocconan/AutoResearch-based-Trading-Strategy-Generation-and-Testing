#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_1D_Trend_v1
Hypothesis: Use Ichimoku cloud on 1d for trend direction and TK cross on 6h for entry timing.
Long when price above 1d cloud, TK line crosses above Kijun on 6h, and volume > 1.5x average.
Short when price below 1d cloud, TK line crosses below Kijun on 6h, and volume > 1.5x average.
Session filter: 08:00-20:00 UTC to avoid low-liquidity periods.
This combines multi-timeframe trend filtering with momentum entry to reduce false signals and control trade frequency.
"""
name = "6h_Ichimoku_Cloud_Filter_1D_Trend_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for Ichimoku cloud (trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + low) / 2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(2)
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    senkou_span_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                      pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2).shift(2)
    
    # Align Ichimoku components to 6h
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Get 6h data for TK cross (entry signal)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 9:
        return np.zeros(n)
    
    # Calculate Tenkan-sen and Kijun-sen on 6h for TK cross
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    tenkan_sen_6h = (pd.Series(high_6h).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low_6h).rolling(window=9, min_periods=9).min()) / 2
    kijun_sen_6h = (pd.Series(high_6h).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low_6h).rolling(window=26, min_periods=26).min()) / 2
    tenkan_sen_6h_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen_6h.values)
    kijun_sen_6h_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen_6h.values)
    
    # Volume filter: current volume > 1.5 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(52, 26, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(tenkan_sen_6h_aligned[i]) or np.isnan(kijun_sen_6h_aligned[i]) or
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 24 bars between trades (4 days on 6h TF) to reduce frequency
            if bars_since_exit < 24:
                continue
                
            # Check session filter
            if not session_filter[i]:
                continue
            
            # Determine cloud boundaries
            cloud_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
            cloud_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
            
            # Long: price above cloud, TK cross bullish, volume filter
            if (close[i] > cloud_top and 
                tenkan_sen_6h_aligned[i] > kijun_sen_6h_aligned[i] and
                tenkan_sen_6h_aligned[i-1] <= kijun_sen_6h_aligned[i-1] and  # crossed above
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price below cloud, TK cross bearish, volume filter
            elif (close[i] < cloud_bottom and 
                  tenkan_sen_6h_aligned[i] < kijun_sen_6h_aligned[i] and
                  tenkan_sen_6h_aligned[i-1] >= kijun_sen_6h_aligned[i-1] and  # crossed below
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price crosses opposite TK line or price re-enters cloud
            if position == 1:
                if (tenkan_sen_6h_aligned[i] < kijun_sen_6h_aligned[i] or  # TK cross bearish
                    close[i] < cloud_top):  # price re-enters cloud
                    signals[i] = 0.0
                    position = 0
                    bars_since_exit = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if (tenkan_sen_6h_aligned[i] > kijun_sen_6h_aligned[i] or  # TK cross bullish
                    close[i] > cloud_bottom):  # price re-enters cloud
                    signals[i] = 0.0
                    position = 0
                    bars_since_exit = 0
                else:
                    signals[i] = -0.25
    
    return signals