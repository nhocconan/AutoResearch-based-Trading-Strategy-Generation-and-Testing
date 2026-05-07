#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_1dTrend_12hVolume
Hypothesis: On 6h timeframe, use daily Ichimoku cloud as trend filter (price above cloud = bullish, below = bearish) and enter on Tenkan-Kijun cross in trend direction. Use 12h volume spike for confirmation. Designed to work in both bull and bear markets by using Ichimoku cloud for dynamic support/resistance and trend direction, reducing whipsaws. Target: 50-150 total trades over 4 years.
"""
name = "6h_Ichimoku_Cloud_Filter_1dTrend_12hVolume"
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
    
    # Get daily data for Ichimoku calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high_daily).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_daily).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high_daily).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_daily).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high_daily).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_daily).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2.0
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_daily, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_daily, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_daily, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_daily, senkou_span_b)
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h volume average
    volume_12h = df_12h['volume'].values
    vol_avg_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_avg_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_12h)
    
    # Volume filter: current volume > 1.8 * 12h average volume
    volume_filter = volume > (vol_avg_12h_aligned * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = 52  # Ensure sufficient warmup for Ichimoku
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data is not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(vol_avg_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        if position == 0:
            # Minimum 8 bars between trades to reduce frequency (6h timeframe)
            if bars_since_entry < 8:
                continue
                
            # Determine trend: price above/below cloud
            upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
            lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
            
            # Long: bullish TK cross + price above cloud + volume filter
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1] and
                close[i] > upper_cloud and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: bearish TK cross + price below cloud + volume filter
            elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                  tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1] and
                  close[i] < lower_cloud and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position != 0:
            # Exit: TK cross in opposite direction
            if position == 1:
                if tenkan_sen_aligned[i] < kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if tenkan_sen_aligned[i] > kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals