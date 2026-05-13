#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Trend_Filtered_Breakout
# Hypothesis: Uses Ichimoku cloud from 1d timeframe as a trend filter and 6h price breaks above/below the cloud for entry.
# In bull markets, price tends to stay above the cloud; breaks above cloud reinforce uptrend.
# In bear markets, price stays below cloud; breaks below cloud reinforce downtrend.
# The cloud acts as dynamic support/resistance, reducing false breakouts.
# Entry: Price breaks above/below 1d Ichimoku cloud with volume confirmation.
# Exit: Price re-enters the cloud.
# Target: 15-35 trades/year per symbol to minimize fee drag.

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

    # Get 1d data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components: Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (52 displacement)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
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
    
    # The cloud is between Senkou Span A and B
    # For cloud top/bottom, we need to shift these forward by 26 periods
    # But for simplicity in filtering, we use current Senkou Span A/B as cloud boundaries
    # (This is a simplification; actual cloud is plotted 26 periods ahead)
    # We'll use the current Senkou Span A and B to determine if price is above/below cloud
    ichimoku_cloud_top = np.maximum(senkou_span_a, senkou_span_b)
    ichimoku_cloud_bottom = np.minimum(senkou_span_a, senkou_span_b)
    
    # Align 1d Ichimoku cloud to 6h timeframe
    ichimoku_cloud_top_aligned = align_htf_to_ltf(prices, df_1d, ichimoku_cloud_top)
    ichimoku_cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, ichimoku_cloud_bottom)
    
    # Volume spike: volume > 2.0 * 20-period average (~5 days worth at 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(ichimoku_cloud_top_aligned[i]) or 
            np.isnan(ichimoku_cloud_bottom_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above cloud + volume spike
            if close[i] > ichimoku_cloud_top_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below cloud + volume spike
            elif close[i] < ichimoku_cloud_bottom_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters cloud (below cloud top)
            if close[i] < ichimoku_cloud_top_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters cloud (above cloud bottom)
            if close[i] > ichimoku_cloud_bottom_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals