#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1w Ichimoku cloud filter and 1d volume confirmation.
Trade breakouts of 12-period Donchian channels aligned with weekly Ichimoku cloud direction.
Use volume spike (>2x 24-period average) to confirm momentum.
Designed to work in bull markets via trend-following breakouts and in bear via mean-reversion at cloud boundaries.
Target: 50-150 total trades over 4 years (12-37/year).
"""
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
    
    # Get 1w data for Ichimoku cloud
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    high_9 = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high_9 + low_9) / 2
    
    high_26 = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high_26 + low_26) / 2
    
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    high_52 = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (high_52 + low_52) / 2
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=24, min_periods=24).mean().values
    
    # Get 6h data for Donchian channels (12-period)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    high_max_12 = pd.Series(high_6h).rolling(window=12, min_periods=12).max().values
    low_min_12 = pd.Series(low_6h).rolling(window=12, min_periods=12).min().values
    
    # Align all to 6h
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b)
    high_max_12_aligned = align_htf_to_ltf(prices, df_6h, high_max_12)
    low_min_12_aligned = align_htf_to_ltf(prices, df_6h, low_min_12)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume filter: current volume > 2x 24-period average
    volume_filter = volume > (vol_ma_1d_aligned * 2.0)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(high_max_12_aligned[i]) or np.isnan(low_min_12_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud color and position
        green_cloud = senkou_span_a_aligned[i] > senkou_span_b_aligned[i]
        red_cloud = senkou_span_a_aligned[i] < senkou_span_b_aligned[i]
        above_cloud = close[i] > max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        below_cloud = close[i] < min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # Long: price breaks above Donchian high, above cloud, volume spike
            if (close[i] > high_max_12_aligned[i] and above_cloud and green_cloud and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, below cloud, volume spike
            elif (close[i] < low_min_12_aligned[i] and below_cloud and red_cloud and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low or enters red cloud
            if close[i] < low_min_12_aligned[i] or red_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high or enters green cloud
            if close[i] > high_max_12_aligned[i] or green_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1wIchimoku_Cloud_Donchian12_VolumeFilter"
timeframe = "6h"
leverage = 1.0