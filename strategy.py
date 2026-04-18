#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_TK_Cross
Hypothesis: Ichimoku Kijun/Tenkan cross with cloud twist (Senkou A/B cross) on daily timeframe acts as a robust trend change signal. Enter on 6h Tenkan/Kijun cross when price is above/below daily cloud and cloud is bullish/bearish. Works in bull via trend continuation and bear via counter-trend reversals at cloud twists.
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
    
    # Get daily data for Ichimoku (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    def ichimoku_components(high, low, close):
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
        period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
        tenkan = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
        period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
        kijun = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods
        senkou_a = ((tenkan + kijun) / 2)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods
        period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
        period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
        senkou_b = ((period52_high + period52_low) / 2)
        
        # Chikou Span (Lagging Span): close shifted -22 periods (not used for signals)
        
        return tenkan, kijun, senkou_a, senkou_b
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = ichimoku_components(high_1d, low_1d, close_1d)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Calculate Ichimoku on 6h for entry signals
    tenkan_6h, kijun_6h, _, _ = ichimoku_components(high, low, close)
    
    # Cloud twist detection: Senkou A/B cross (trend change signal)
    # Bullish twist: Senkou A crosses above Senkou B
    # Bearish twist: Senkou A crosses below Senkou B
    senkou_a_prev = np.roll(senkou_a_1d_aligned, 1)
    senkou_b_prev = np.roll(senkou_b_1d_aligned, 1)
    senkou_a_prev[0] = np.nan
    senkou_b_prev[0] = np.nan
    
    bullish_twist = (senkou_a_1d_aligned > senkou_b_1d_aligned) & (senkou_a_prev <= senkou_b_prev)
    bearish_twist = (senkou_a_1d_aligned < senkou_b_1d_aligned) & (senkou_a_prev >= senkou_b_prev)
    
    # Tenkan/Kijun cross signals
    tenkan_6h_prev = np.roll(tenkan_6h, 1)
    kijun_6h_prev = np.roll(kijun_6h, 1)
    tenkan_6h_prev[0] = np.nan
    kijun_6h_prev[0] = np.nan
    
    tk_bullish_cross = (tenkan_6h > kijun_6h) & (tenkan_6h_prev <= kijun_6h_prev)
    tk_bearish_cross = (tenkan_6h < kijun_6h) & (tenkan_6h_prev >= kijun_6h_prev)
    
    # Price position relative to cloud
    price_above_cloud = close > np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    price_below_cloud = close < np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    
    signals = np.zeros(n)
    
    # Start after warmup period for Ichimoku calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any values are NaN
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(bullish_twist[i]) or np.isnan(bearish_twist[i])):
            continue
            
        # Bullish setup: price above cloud AND bullish cloud twist OR TK bullish cross in bullish regime
        if price_above_cloud[i]:
            if bullish_twist[i] or tk_bullish_cross[i]:
                signals[i] = 0.25
        
        # Bearish setup: price below cloud AND bearish cloud twist OR TK bearish cross in bearish regime
        elif price_below_cloud[i]:
            if bearish_twist[i] or tk_bearish_cross[i]:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_TK_Cross"
timeframe = "6h"
leverage = 1.0