#!/usr/bin/env python3
# 6h_1d_Ichimoku_Kumo_Twist_With_Volume_Filter
# Hypothesis: Uses Ichimoku cloud from 1d timeframe for trend direction and twist detection.
# Enters long when TK line crosses above Kijun-sen in bullish cloud (price > cloud) with volume confirmation.
# Enters short when TK line crosses below Kijun-sen in bearish cloud (price < cloud) with volume confirmation.
# The Kumo twist (Senkou Span A/B cross) confirms trend strength. Designed for low trade frequency
# to avoid fee drag, works in both bull and bear markets via cloud filter.

name = "6h_1d_Ichimoku_Kumo_Twist_With_Volume_Filter"
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
    
    # 1d data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Ichimoku
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components
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
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # Not used in signals but calculated for completeness
    
    # Kumo twist: Senkou Span A crosses above/below Senkou Span B
    # Bullish twist: Senkou Span A > Senkou Span B (future cloud bullish)
    # Bearish twist: Senkou Span A < Senkou Span B (future cloud bearish)
    kumo_twist_bullish = senkou_span_a > senkou_span_b
    kumo_twist_bearish = senkou_span_a < senkou_span_b
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    kumo_twist_bullish_aligned = align_htf_to_ltf(prices, df_1d, kumo_twist_bullish.astype(float))
    kumo_twist_bearish_aligned = align_htf_to_ltf(prices, df_1d, kumo_twist_bearish.astype(float))
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TK crosses above Kijun in bullish cloud with bullish twist and volume
            tk_cross_above = (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                              tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1])
            price_above_cloud = close[i] > cloud_top[i]
            bullish_alignment = kumo_twist_bullish_aligned[i] and price_above_cloud
            
            if tk_cross_above and bullish_alignment and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            
            # Short: TK crosses below Kijun in bearish cloud with bearish twist and volume
            elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                  tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]):
                tk_cross_below = True
                price_below_cloud = close[i] < cloud_bottom[i]
                bearish_alignment = kumo_twist_bearish_aligned[i] and price_below_cloud
                
                if tk_cross_below and bearish_alignment and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: TK crosses below Kijun or price enters cloud
            tk_cross_below = (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                              tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1])
            price_in_cloud = close[i] <= cloud_top[i] and close[i] >= cloud_bottom[i]
            
            if tk_cross_below or price_in_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: TK crosses above Kijun or price enters cloud
            tk_cross_above = (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                              tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1])
            price_in_cloud = close[i] <= cloud_top[i] and close[i] >= cloud_bottom[i]
            
            if tk_cross_above or price_in_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals