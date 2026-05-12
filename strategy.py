#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Breakout_Trend
# Hypothesis: Use Ichimoku Cloud from daily timeframe for trend direction and 6h for entry timing. 
# Enter long when price breaks above Kumo (cloud) with bullish TK cross (Tenkan > Kijun) and price > Senkou Span A.
# Enter short when price breaks below Kumo with bearish TK cross and price < Senkou Span B.
# Exit when price re-enters the cloud or TK cross reverses.
# Ichimoku provides multi-line trend/filter system that works in both bull/bear markets by adapting to price action.
# Targets 15-30 trades/year on 6h to minimize fee drag.

name = "6h_Ichimoku_Cloud_Breakout_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan-sen, Kijun-sen, Senkou Span A/B, Chikou Span"""
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
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for Ichimoku trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Ichimoku on daily data
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align Ichimoku components to 6h timeframe (wait for daily close)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if any required value is NaN
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Determine cloud boundaries and TK cross
        senkou_top = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        senkou_bottom = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        tk_bullish = tenkan_1d_aligned[i] > kijun_1d_aligned[i]
        tk_bearish = tenkan_1d_aligned[i] < kijun_1d_aligned[i]

        if position == 0:
            # LONG: Price breaks above cloud + bullish TK cross + volume spike
            if (close[i] > senkou_top and 
                tk_bullish and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below cloud + bearish TK cross + volume spike
            elif (close[i] < senkou_bottom and 
                  tk_bearish and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters cloud or TK turns bearish
            if (close[i] < senkou_top or not tk_bullish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters cloud or TK turns bullish
            if (close[i] > senkou_bottom or tk_bullish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals