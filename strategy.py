#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist
Hypothesis: Ichimoku Cloud Twist (TK cross + cloud color change) from daily timeframe,
combined with 6h price action above/below cloud and volume confirmation.
Works in bull/bear markets: cloud acts as dynamic support/resistance.
Target: 15-30 trades/year on 6h to avoid fee drag.
"""

name = "6h_Ichimoku_Kumo_Twist"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind
    chikou = close_1d  # We'll handle alignment properly
    
    # Align all Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    chikou_6h = align_htf_to_ltf(prices, df_1d, chikou)
    
    # Get 6h price and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Ichimoku calculation (52 periods for Senkou B) + alignment delay
    start_idx = 52 + 26  # 52 for calculation + 26 for Senkou shift alignment
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(tenkan_6h[i]) or 
            np.isnan(kijun_6h[i]) or
            np.isnan(senkou_a_6h[i]) or
            np.isnan(senkou_b_6h[i]) or
            np.isnan(chikou_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Cloud top and bottom
        cloud_top = np.maximum(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom = np.minimum(senkou_a_6h[i], senkou_b_6h[i])
        
        # Cloud color: green (bullish) when Senkou A > Senkou B
        cloud_bullish = senkou_a_6h[i] > senkou_b_6h[i]
        
        # TK Cross: Tenkan > Kijun (bullish cross)
        tk_bullish = tenkan_6h[i] > kijun_6h[i]
        
        # Chikou confirmation: Chikou above price 26 periods ago
        chikou_confirm = chikou_6h[i] > close[i - 26] if i >= 26 else False
        
        if position == 0:
            # Long: TK bullish cross + price above cloud + bullish cloud + volume
            if tk_bullish and close[i] > cloud_top and cloud_bullish and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: TK bearish cross + price below cloud + bearish cloud + volume
            elif not tk_bullish and close[i] < cloud_bottom and not cloud_bullish and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TK bearish cross OR price below cloud OR cloud turns bearish
            if (not tk_bullish) or (close[i] < cloud_top) or (not cloud_bullish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TK bullish cross OR price above cloud OR cloud turns bullish
            if tk_bullish or (close[i] > cloud_bottom) or cloud_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals