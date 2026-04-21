#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_Cloud_Filter
Hypothesis: Use Ichimoku cloud (Tenkan/Kijun cross and price above/below cloud) on 1d timeframe for trend direction, with entry on 6h timeframe when price crosses Tenkan-Kijun line on 6h in direction of 1d trend. Exit when price returns to Kijun line on 6h. Ichimoku provides built-in trend, momentum, and support/resistance, working in both bull and bear markets by filtering trades with higher timeframe cloud color.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan, Kijun, Senkou A/B, Chikou"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 52 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align 1d Ichimoku components to 6h
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Calculate Ichimoku on 6h for entry signals
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tenkan_6h, kijun_6h, _, _ = calculate_ichimoku(high, low, close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after warmup for Ichimoku
        # Skip if indicators not ready
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        
        # 1d trend filter: price above/below cloud
        # Green cloud: senkou_a > senkou_b (bullish)
        # Red cloud: senkou_a < senkou_b (bearish)
        is_bullish_cloud = senkou_a_1d_aligned[i] > senkou_b_1d_aligned[i]
        is_bearish_cloud = senkou_a_1d_aligned[i] < senkou_b_1d_aligned[i]
        
        # 6h Tenkan-Kijun cross
        tk_cross_up = tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]
        tk_cross_down = tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]
        
        if position == 0:
            # Long: Price above cloud (bullish) + TK cross up on 6h
            if price_close > senkou_a_1d_aligned[i] and price_close > senkou_b_1d_aligned[i] and tk_cross_up:
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud (bearish) + TK cross down on 6h
            elif price_close < senkou_a_1d_aligned[i] and price_close < senkou_b_1d_aligned[i] and tk_cross_down:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price crosses Kijun line on 6h (reversal signal)
            if position == 1 and tenkan_6h[i] < kijun_6h[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and tenkan_6h[i] > kijun_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Filter"
timeframe = "6h"
leverage = 1.0