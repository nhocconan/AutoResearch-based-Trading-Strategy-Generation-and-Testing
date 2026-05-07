#!/usr/bin/env python3
# 4h_12h_Ichimoku_Trend_Continuation
# Hypothesis: Uses Ichimoku Cloud from 12h timeframe for trend identification and 4h for entry timing.
# Enters long when price is above 12h Kumo (cloud) and Tenkan crosses above Kijun on 4h with volume confirmation.
# Enters short when price is below 12h Kumo and Tenkan crosses below Kijun on 4h with volume confirmation.
# Ichimoku provides robust trend definition that works in both bull and bear markets; the 4h cross signals
# momentum continuation within the trend; volume confirms breakout strength. Targets 20-40 trades/year.

name = "4h_12h_Ichimoku_Trend_Continuation"
timeframe = "4h"
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
    
    # Get 12h data for Ichimoku (HTF as specified)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Ichimoku components on 12h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_9 = pd.Series(high_12h).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_12h).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_12h = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_26 = pd.Series(high_12h).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_12h).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_12h = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a_12h = (tenkan_12h + kijun_12h) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_52 = pd.Series(high_12h).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_12h).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b_12h = (max_high_52 + min_low_52) / 2
    
    # Align Ichimoku components to 4h timeframe
    tenkan_4h = align_htf_to_ltf(prices, df_12h, tenkan_12h)
    kijun_4h = align_htf_to_ltf(prices, df_12h, kijun_12h)
    senkou_span_a_4h = align_htf_to_ltf(prices, df_12h, senkou_span_a_12h)
    senkou_span_b_4h = align_htf_to_ltf(prices, df_12h, senkou_span_b_12h)
    
    # Kumo (Cloud) boundaries: Senkou Span A and B
    kumo_top_4h = np.maximum(senkou_span_a_4h, senkou_span_b_4h)
    kumo_bottom_4h = np.minimum(senkou_span_a_4h, senkou_span_b_4h)
    
    # Calculate 4h Tenkan and Kijun for crossover signals
    max_high_9_4h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    min_low_9_4h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_4h_fast = (max_high_9_4h + min_low_9_4h) / 2
    
    max_high_26_4h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    min_low_26_4h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_4h_fast = (max_high_26_4h + min_low_26_4h) / 2
    
    # Volume spike filter on 4h (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(52, n):
        # Skip if any critical value is NaN
        if (np.isnan(kumo_top_4h[i]) or np.isnan(kumo_bottom_4h[i]) or 
            np.isnan(tenkan_4h_fast[i]) or np.isnan(kijun_4h_fast[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: Price above Kumo, Tenkan crosses above Kijun, volume spike
            if (close[i] > kumo_top_4h[i] and 
                tenkan_4h_fast[i] > kijun_4h_fast[i] and 
                tenkan_4h_fast[i-1] <= kijun_4h_fast[i-1] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: Price below Kumo, Tenkan crosses below Kijun, volume spike
            elif (close[i] < kumo_bottom_4h[i] and 
                  tenkan_4h_fast[i] < kijun_4h_fast[i] and 
                  tenkan_4h_fast[i-1] >= kijun_4h_fast[i-1] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit: price breaks below Kumo bottom or Tenkan crosses below Kijun
            if (close[i] < kumo_bottom_4h[i] or 
                (tenkan_4h_fast[i] < kijun_4h_fast[i] and tenkan_4h_fast[i-1] >= kijun_4h_fast[i-1])):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Kumo top or Tenkan crosses above Kijun
            if (close[i] > kumo_top_4h[i] or 
                (tenkan_4h_fast[i] > kijun_4h_fast[i] and tenkan_4h_fast[i-1] <= kijun_4h_fast[i-1])):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals