#!/usr/bin/env python3
# 6h_1d_Ichimoku_Kumo_Twist_VolumeFilter
# Hypothesis: Ichimoku Kumo (cloud) twist on daily timeframe combined with volume confirmation on 6h timeframe.
# The Kumo twist (Senkou Span A/B crossover) signals trend changes. We enter long when price is above cloud
# and Tenkan-sen crosses above Kijun-sen, with volume confirmation. Short when price below cloud and Tenkan
# crosses below Kijun. Uses daily Ichimoku for trend and 6h for entry timing and volume filter.
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position size.

name = "6h_1d_Ichimoku_Kumo_Twist_VolumeFilter"
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
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)  # shifted 26 periods ahead
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = ((pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                      pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2).shift(26)
    
    # Kumo twist detection: Senkou Span A crosses above/below Senkou Span B
    # Kumo twist bullish: Senkou Span A crosses above Senkou Span B
    # Kumo twist bearish: Senkou Span A crosses below Senkou Span B
    senkou_span_a_values = senkou_span_a.values
    senkou_span_b_values = senkou_span_b.values
    
    # Kumo twist signals (1 for bullish twist, -1 for bearish twist, 0 otherwise)
    kumo_twist = np.zeros_like(senkou_span_a_values)
    # Bullish twist: Senkou A crosses above Senkou B
    bullish_twist = (senkou_span_a_values[1:] > senkou_span_b_values[1:]) & (senkou_span_a_values[:-1] <= senkou_span_b_values[:-1])
    # Bearish twist: Senkou A crosses below Senkou B
    bearish_twist = (senkou_span_a_values[1:] < senkou_span_b_values[1:]) & (senkou_span_a_values[:-1] >= senkou_span_b_values[:-1])
    kumo_twist[1:] = np.where(bullish_twist, 1, np.where(bearish_twist, -1, 0))
    
    # Current Kumo cloud (for price position relative to cloud)
    # Senkou Span A and B shifted forward by 26 periods, so current cloud is from 26 periods ago
    senkou_span_a_current = senkou_span_a_values  # already shifted in calculation
    senkou_span_b_current = senkou_span_b_values  # already shifted in calculation
    
    # Align daily Ichimoku components to 6h timeframe
    kumo_twist_aligned = align_htf_to_ltf(prices, df_1d, kumo_twist)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_current)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_current)
    
    # Volume spike detection on 6h timeframe
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values  # 20*6h = 5 days
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kumo_twist_aligned[i]) or np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or np.isnan(senkou_span_a_aligned[i]) or 
            np.isnan(senkou_span_b_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5 * 20-period average volume
        volume_spike = volume[i] > 1.5 * volume_ma[i]
        
        # Price position relative to cloud
        above_cloud = close[i] > max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        below_cloud = close[i] < min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Tenkan/Kijun crossover
        tenkan_crosses_kijun = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tenkan_crosses_below_kijun = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        if position == 0:
            # Long: bullish Kumo twist OR (price above cloud AND Tenkan crosses above Kijun) with volume
            if (kumo_twist_aligned[i] == 1 or 
                (above_cloud and tenkan_crosses_kijun and volume_spike)):
                signals[i] = 0.25
                position = 1
            # Short: bearish Kumo twist OR (price below cloud AND Tenkan crosses below Kijun) with volume
            elif (kumo_twist_aligned[i] == -1 or 
                  (below_cloud and tenkan_crosses_below_kijun and volume_spike)):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if bearish Kumo twist OR price falls below cloud OR Tenkan crosses below Kijun
            if (kumo_twist_aligned[i] == -1 or 
                below_cloud or 
                tenkan_crosses_below_kijun):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if bullish Kumo twist OR price rises above cloud OR Tenkan crosses above Kijun
            if (kumo_twist_aligned[i] == 1 or 
                above_cloud or 
                tenkan_crosses_kijun):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals