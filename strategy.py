#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Twist_V1
Strategy: Ichimoku Cloud twist on 6h with daily Cloud color filter.
- Uses TK line cross (Tenkan/Kijun) as entry trigger
- Requires price to be above/below cloud (Senkou Span A/B) for confirmation
- Cloud color (Senkou A > Senkou B = bullish, < = bearish) from daily timeframe
- Exit when TK cross reverses or price re-enters cloud
Position size: 0.25
Designed to capture trend changes with Ichimoku's multi-line confirmation system.
Timeframe: 6h
"""

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
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26  # Kijun period for Senkou span displacement
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                     pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # Displace Senkou spans forward by Kijun period (26 periods)
    # We'll handle displacement in alignment - for now calculate raw values
    
    # Get daily data for Cloud color filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Ichimoku components for cloud color
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily Tenkan and Kijun
    tenkan_sen_1d = (pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                     pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    kijun_sen_1d = (pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                    pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Daily Senkou Span A and B
    senkou_span_a_1d = ((tenkan_sen_1d + kijun_sen_1d) / 2)
    senkou_span_b_1d = (pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                        pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # Cloud color: Senkou A > Senkou B = bullish cloud, Senkou A < Senkou B = bearish cloud
    cloud_bullish = senkou_span_a_1d > senkou_span_b_1d
    cloud_bearish = senkou_span_a_1d < senkou_span_b_1d
    
    # Align daily Ichimoku components to 6h timeframe
    tenkan_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen_1d.values)
    kijun_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_1d.values)
    senkou_span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d.values)
    senkou_span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d.values)
    cloud_bullish_aligned = align_htf_to_ltf(prices, df_1d, cloud_bullish.astype(float))
    cloud_bearish_aligned = align_htf_to_ltf(prices, df_1d, cloud_bearish.astype(float))
    
    # Align 6h Ichimoku components (no displacement needed for TK cross)
    tenkan_sen_aligned = tenkan_sen.values  # Already on 6h timeframe
    kijun_sen_aligned = kijun_sen.values
    senkou_span_a_aligned = senkou_span_a.values
    senkou_span_b_aligned = senkou_span_b.values
    
    # Displace Senkou spans forward by 26 periods for cloud visualization
    # For trading, we use current Senkou spans to determine if price is above/below cloud
    # The displacement is already accounted for in the calculation method
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(kijun_period, senkou_span_b_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_sen_1d_aligned[i]) or np.isnan(kijun_sen_1d_aligned[i]) or 
            np.isnan(senkou_span_a_1d_aligned[i]) or np.isnan(senkou_span_b_1d_aligned[i]) or
            np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        # TK Cross signals
        tk_cross_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]
        tk_cross_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]
        
        # Price relative to cloud (using current Senkou spans)
        price_above_cloud = close[i] > max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        price_below_cloud = close[i] < min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Daily cloud color
        daily_cloud_bullish = cloud_bullish_aligned[i] > 0.5
        daily_cloud_bearish = cloud_bearish_aligned[i] > 0.5
        
        # Entry conditions
        if position == 0:
            # Long: Bullish TK cross + price above cloud + bullish daily cloud
            if tk_cross_bullish and price_above_cloud and daily_cloud_bullish:
                signals[i] = 0.25
                position = 1
            # Short: Bearish TK cross + price below cloud + bearish daily cloud
            elif tk_cross_bearish and price_below_cloud and daily_cloud_bearish:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bearish TK cross OR price re-enters cloud OR daily cloud turns bearish
            if tk_cross_bearish or not price_above_cloud or not daily_cloud_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bullish TK cross OR price re-enters cloud OR daily cloud turns bullish
            if tk_cross_bullish or not price_below_cloud or not daily_cloud_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Twist_V1"
timeframe = "6h"
leverage = 1.0