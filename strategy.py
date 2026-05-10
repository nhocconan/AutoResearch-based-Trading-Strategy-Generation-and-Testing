#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend
Hypothesis: Price crosses the Kumo (cloud) twist on 6h chart, with 1d Ichimoku trend filter.
Kumo twist (Senkou Span A/B cross) signals potential trend reversals. Trading in the direction
of the 1d Ichimoku trend (price above/below Kumo) ensures alignment with higher timeframe.
Works in bull/bear by only taking trades aligned with 1d trend. Target: 15-25 trades/year (60-100 total).
"""

name = "6h_Ichimoku_Kumo_Twist_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku parameters
    tenkan = 9
    kijun = 26
    senkou = 52
    
    # Calculate 6h Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.max(arr[i - window + 1:i + 1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.min(arr[i - window + 1:i + 1])
        return res
    
    # Tenkan and Kijun for 6h
    tenkan_sen = (rolling_max(high, tenkan) + rolling_min(low, tenkan)) / 2
    kijun_sen = (rolling_max(high, kijun) + rolling_min(low, kijun)) / 2
    
    # Senkou Span A and B (leading span)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    senkou_span_b = (rolling_max(high, senkou) + rolling_min(low, senkou)) / 2
    
    # Chikou Span (lagging span) - not used for entry but for confirmation
    chikou_span = np.roll(close, -kijun)  # shifted back by kijun periods
    
    # Kumo twist detection: Senkou Span A crosses Senkou Span B
    # Bullish twist: Senkou A crosses above Senkou B
    # Bearish twist: Senkou A crosses below Senkou B
    senkou_a_above_b = senkou_span_a > senkou_span_b
    senkou_a_below_b = senkou_span_a < senkou_span_b
    
    # Kumo twist signals (crossovers)
    kumo_twist_bull = senkou_a_above_b & ~np.roll(senkou_a_above_b, 1)
    kumo_twist_bear = senkou_a_below_b & ~np.roll(senkou_a_below_b, 1)
    
    # Price relative to Kumo (cloud)
    kumo_top = np.maximum(senkou_span_a, senkou_span_b)
    kumo_bottom = np.minimum(senkou_span_a, senkou_span_b)
    price_above_kumo = close > kumo_top
    price_below_kumo = close < kumo_bottom
    
    # Get 1d Ichimoku for trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Ichimoku components
    tenkan_sen_1d = (rolling_max(high_1d, tenkan) + rolling_min(low_1d, tenkan)) / 2
    kijun_sen_1d = (rolling_max(high_1d, kijun) + rolling_min(low_1d, kijun)) / 2
    senkou_span_a_1d = ((tenkan_sen_1d + kijun_sen_1d) / 2)
    senkou_span_b_1d = (rolling_max(high_1d, senkou) + rolling_min(low_1d, senkou)) / 2
    kumo_top_1d = np.maximum(senkou_span_a_1d, senkou_span_b_1d)
    kumo_bottom_1d = np.minimum(senkou_span_a_1d, senkou_span_b_1d)
    
    # 1d trend: price above/below 1d Kumo
    price_above_kumo_1d = close_1d > kumo_top_1d
    price_below_kumo_1d = close_1d < kumo_bottom_1d
    
    # Align 1d Ichimoku to 6h
    price_above_kumo_1d_aligned = align_htf_to_ltf(prices, df_1d, price_above_kumo_1d.astype(float))
    price_below_kumo_1d_aligned = align_htf_to_ltf(prices, df_1d, price_below_kumo_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = senkou  # Wait for Senkou Span calculation
    
    for i in range(start_idx, n):
        if np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check for Kumo twist on 6h
        bullish_twist = kumo_twist_bull[i] if i < len(kumo_twist_bull) else False
        bearish_twist = kumo_twist_bear[i] if i < len(kumo_twist_bear) else False
        
        if position == 0:
            # Long: bullish Kumo twist + price above 6h Kumo + 1d uptrend
            if bullish_twist and price_above_kumo[i] and price_above_kumo_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish Kumo twist + price below 6h Kumo + 1d downtrend
            elif bearish_twist and price_below_kumo[i] and price_below_kumo_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: bearish Kumo twist or price falls below 6h Kumo
            if bearish_twist or not price_above_kumo[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: bullish Kumo twist or price rises above 6h Kumo
            if bullish_twist or not price_below_kumo[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals