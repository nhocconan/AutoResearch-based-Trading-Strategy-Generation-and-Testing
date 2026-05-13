#!/usr/bin/env python3
# 1d_Weekly_Ichimoku_Cloud_Turn
# Hypothesis: Weekly Ichimoku Cloud identifies major trend shifts; price crossing above/below cloud with weekly TK cross confirms momentum. Daily price must be outside cloud to avoid whipsaws. Uses volume confirmation (>1.5x 20-day avg) to filter low-conviction moves. Designed for low trade frequency (<15/year) to minimize fee drag and improve generalization in both bull and bear markets.

name = "1d_Weekly_Ichimoku_Cloud_Turn"
timeframe = "1d"
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

    # Ichimoku components on weekly data
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2

    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2

    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)

    # Chikou Span (Lagging Span): close shifted 26 periods back
    chikou = np.roll(close_1w, 26)
    chikou[:26] = np.nan

    # Align all components to daily timeframe (already shifted for look-ahead avoidance)
    tenkan_aligned = align_htf_to_ltf(prices, df_1w, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1w, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_b)
    chikou_aligned = align_htf_to_ltf(prices, df_1w, chikou)

    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)

    # TK Cross: Tenkan crosses Kijun
    tk_cross = np.where((tenkan_aligned > kijun_aligned) & (np.roll(tenkan_aligned, 1) <= np.roll(kijun_aligned, 1)), 1,
                        np.where((tenkan_aligned < kijun_aligned) & (np.roll(tenkan_aligned, 1) >= np.roll(kijun_aligned, 1)), -1, 0))

    # Volume confirmation: >1.5x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if any required value is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(tk_cross[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above cloud + bullish TK cross + volume confirmation
            if (close[i] > cloud_top[i] and
                tk_cross[i] == 1 and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below cloud + bearish TK cross + volume confirmation
            elif (close[i] < cloud_bottom[i] and
                  tk_cross[i] == -1 and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below cloud or bearish TK cross
            if (close[i] < cloud_top[i] or tk_cross[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above cloud or bullish TK cross
            if (close[i] > cloud_bottom[i] or tk_cross[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals