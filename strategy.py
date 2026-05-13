#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Trend_Filtered_Breakout
# Hypothesis: Ichimoku Cloud from 1d timeframe provides trend direction and support/resistance.
# Go long when price breaks above Kumo (cloud) top with bullish TK cross and volume confirmation.
# Go short when price breaks below Kumo bottom with bearish TK cross and volume confirmation.
# Uses 6h for entry timing and 1d for Ichimoku filter to reduce false signals.
# Works in bull markets (bullish TK cross + price above cloud) and bear markets (bearish TK cross + price below cloud).
# Target: 12-30 trades/year per symbol to minimize fee drag.

name = "6h_Ichimoku_Cloud_Trend_Filtered_Breakout"
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

    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components
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
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2, shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Chikou Span (Lagging Span): current close shifted 26 periods behind
    chikou = close_1d  # Will be aligned properly
    
    # Kumo (Cloud) top and bottom
    kumo_top = np.maximum(senkou_a, senkou_b)
    kumo_bottom = np.minimum(senkou_a, senkou_b)
    
    # TK Cross signals
    tk_bullish = tenkan > kijun
    tk_bearish = tenkan < kijun
    
    # Align 1d Ichimoku components to 6h timeframe
    kumo_top_aligned = align_htf_to_ltf(prices, df_1d, kumo_top)
    kumo_bottom_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom)
    tk_bullish_aligned = align_htf_to_ltf(prices, df_1d, tk_bullish)
    tk_bearish_aligned = align_htf_to_ltf(prices, df_1d, tk_bearish)
    chikou_aligned = align_htf_to_ltf(prices, df_1d, chikou)
    
    # Volume spike: volume > 2.0 * 20-period average (approx 5 days at 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(kumo_top_aligned[i]) or 
            np.isnan(kumo_bottom_aligned[i]) or 
            np.isnan(tk_bullish_aligned[i]) or 
            np.isnan(tk_bearish_aligned[i]) or
            np.isnan(chikou_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above Kumo top + bullish TK cross + Chikou above price + volume spike
            if (close[i] > kumo_top_aligned[i] and 
                tk_bullish_aligned[i] and 
                chikou_aligned[i] > close[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below Kumo bottom + bearish TK cross + Chikou below price + volume spike
            elif (close[i] < kumo_bottom_aligned[i] and 
                  tk_bearish_aligned[i] and 
                  chikou_aligned[i] < close[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below Kumo bottom or bearish TK cross
            if close[i] < kumo_bottom_aligned[i] or not tk_bullish_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above Kumo top or bullish TK cross
            if close[i] > kumo_top_aligned[i] or tk_bullish_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals