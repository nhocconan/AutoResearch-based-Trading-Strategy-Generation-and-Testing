#!/usr/bin/env python3
# 6h_Daily_Ichimoku_Kumo_Twist_Trend
# Hypothesis: Ichimoku Kumo (cloud) twist from daily timeframe acts as a strong trend change signal.
# A bullish twist (Senkou Span A crosses above Senkou Span B) signals potential uptrend,
# while a bearish twist signals potential downtrend. We enter on the 6h timeframe when
# price confirms the trend by being above/below the Kumo, with the twist acting as the trigger.
# This captures major trend shifts and works in both bull (bullish twists in uptrends) and
# bear (bearish twists in downtrends) markets. Low frequency due to reliance on daily Ichimoku.

name = "6h_Daily_Ichimoku_Kumo_Twist_Trend"
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

    # Get daily data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components on daily data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Kumo twist signals: bullish when Senkou Span A crosses above Senkou Span B
    # Bearish when Senkou Span A crosses below Senkou Span B
    # We need to detect the cross and then wait for confirmation
    bullish_twist = (senkou_span_a > senkou_span_b) & (np.roll(senkou_span_a, 1) <= np.roll(senkou_span_b, 1))
    bearish_twist = (senkou_span_a < senkou_span_b) & (np.roll(senkou_span_a, 1) >= np.roll(senkou_span_b, 1))
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    bullish_twist_aligned = align_htf_to_ltf(prices, df_1d, bullish_twist.astype(float))
    bearish_twist_aligned = align_htf_to_ltf(prices, df_1d, bearish_twist.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or 
            np.isnan(senkou_span_b_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bullish Kumo twist + price above Kumo
            if bullish_twist_aligned[i] > 0.5 and close[i] > max(senkou_span_a_aligned[i], senkou_span_b_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish Kumo twist + price below Kumo
            elif bearish_twist_aligned[i] > 0.5 and close[i] < min(senkou_span_a_aligned[i], senkou_span_b_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below Kumo or bearish twist occurs
            if close[i] < min(senkou_span_a_aligned[i], senkou_span_b_aligned[i]) or bearish_twist_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above Kumo or bullish twist occurs
            if close[i] > max(senkou_span_a_aligned[i], senkou_span_b_aligned[i]) or bullish_twist_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals