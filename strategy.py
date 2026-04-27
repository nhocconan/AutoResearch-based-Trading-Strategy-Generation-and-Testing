#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_Trend_Continuation
Hypothesis: Uses Ichimoku cloud twist (Senkou Span A/B cross) from daily timeframe as trend filter,
combined with 6h Tenkan-Kijun cross for entry timing. In bull regime (price above cloud, Senkou A > Senkou B),
enter long on Tenkan/Kijun bullish cross. In bear regime (price below cloud, Senkou A < Senkou B),
enter short on Tenkan/Kijun bearish cross. Exit when price re-enters cloud or twist reverses.
Ichimoku twist confirms higher-timeframe trend structure, reducing false breaks in ranging markets.
Designed for 6h timeframe with 50-150 total trades over 4 years (12-37/year) at 0.25 position size.
Works in both bull/bear via cloud/twist regime filter + momentum cross entries.
"""

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
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26  # Senkou spans plotted 26 periods ahead
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max().values +
                  pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min().values) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max().values +
                 pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min().values) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    senkou_span_b = (pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values +
                     pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values) / 2
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind (not needed for this strategy)
    
    # Align Ichimoku components to 6h timeframe (no additional delay needed for Senkou spans as they are already leading)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate Kumo (cloud) twist: Senkou Span A/B cross
    # Kumo twist bullish: Senkou A > Senkou B
    # Kumo twist bearish: Senkou A < Senkou B
    kumo_twist_bullish = senkou_span_a_aligned > senkou_span_b_aligned
    kumo_twist_bearish = senkou_span_a_aligned < senkou_span_b_aligned
    
    # Price above/below cloud
    price_above_cloud = (close > np.maximum(senkou_span_a_aligned, senkou_span_b_aligned))
    price_below_cloud = (close < np.minimum(senkou_span_a_aligned, senkou_span_b_aligned))
    
    # Tenkan/Kijun cross for entry signals
    # Bullish cross: Tenkan crosses above Kijun
    tenkan_kijun_cross_up = (tenkan_aligned > kijun_aligned) & (np.roll(tenkan_aligned, 1) <= np.roll(kijun_aligned, 1))
    # Bearish cross: Tenkan crosses below Kijun
    tenkan_kijun_cross_down = (tenkan_aligned < kijun_aligned) & (np.roll(tenkan_aligned, 1) >= np.roll(kijun_aligned, 1))
    
    # Handle first value for cross signals
    tenkan_kijun_cross_up[0] = False
    tenkan_kijun_cross_down[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need Ichimoku calculations (52 periods for Senkou B)
    start_idx = senkou_span_b_period + displacement  # 52 + 26 = 78
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filters
        bullish_regime = kumo_twist_bullish[i] and price_above_cloud[i]
        bearish_regime = kumo_twist_bearish[i] and price_below_cloud[i]
        
        if position == 0:
            # Enter long in bullish regime on Tenkan/Kijun bullish cross
            if bullish_regime and tenkan_kijun_cross_up[i]:
                signals[i] = size
                position = 1
            # Enter short in bearish regime on Tenkan/Kijun bearish cross
            elif bearish_regime and tenkan_kijun_cross_down[i]:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price re-enters cloud or regime turns bearish
            exit_condition = (not price_above_cloud[i]) or (not kumo_twist_bullish[i])
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price re-enters cloud or regime turns bullish
            exit_condition = (not price_below_cloud[i]) or (not kumo_twist_bearish[i])
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_Trend_Continuation"
timeframe = "6h"
leverage = 1.0