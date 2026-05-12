#!/usr/bin/env python3
"""
1d_Ichimoku_TenkanKijun_Cross_1wTrend
Hypothesis: Ichimoku Cloud system with Tenkan/Kijun cross for entry and weekly trend filter for direction works in both bull and bear markets. In bull markets (price > weekly Kumo), buy when Tenkan crosses above Kijun; in bear markets (price < weekly Kumo), sell when Tenkan crosses below Kijun. Uses Kumo twist (Senkou A/B crossover) for trend confirmation and volume filter to avoid false signals. Designed for low trade frequency (<25/year) to minimize fee drag.
"""

name = "1d_Ichimoku_TenkanKijun_Cross_1wTrend"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Ichimoku components (daily timeframe)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2

    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2

    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2

    # Weekly trend components
    # Weekly Tenkan/Kijun for trend direction
    wk_period9_high = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    wk_period9_low = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    wk_tenkan = (wk_period9_high + wk_period9_low) / 2

    wk_period26_high = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    wk_period26_low = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    wk_kijun = (wk_period26_high + wk_period26_low) / 2

    # Weekly Senkou Span A and B
    wk_senkou_a = (wk_tenkan + wk_kijun) / 2
    wk_period52_high = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    wk_period52_low = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    wk_senkou_b = (wk_period52_high + wk_period52_low) / 2

    # Kumo twist: Senkou A/B crossover (bullish when A > B)
    kumo_twist = senkou_a - senkou_b
    wk_kumo_twist = wk_senkou_a - wk_senkou_b

    # Align weekly data to daily
    wk_tenkan_aligned = align_htf_to_ltf(prices, df_1w, wk_tenkan)
    wk_kijun_aligned = align_htf_to_ltf(prices, df_1w, wk_kijun)
    wk_senkou_a_aligned = align_htf_to_ltf(prices, df_1w, wk_senkou_a)
    wk_senkou_b_aligned = align_htf_to_ltf(prices, df_1w, wk_senkou_b)
    wk_kumo_twist_aligned = align_htf_to_ltf(prices, df_1w, wk_kumo_twist)

    # Volume confirmation: volume > 1.5x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # Start after Senkou B calculation
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(wk_tenkan_aligned[i]) or np.isnan(wk_kijun_aligned[i]) or
            np.isnan(wk_senkou_a_aligned[i]) or np.isnan(wk_senkou_b_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Determine market regime from weekly Kumo
            price_vs_wk_kumo = close[i] - max(wk_senkou_a_aligned[i], wk_senkou_b_aligned[i])
            price_vs_wk_kumo_below = min(wk_senkou_a_aligned[i], wk_senkou_b_aligned[i]) - close[i]
            
            # BULLISH: Price above weekly Kumo + Kumo twist bullish (A > B)
            if (price_vs_wk_kumo > 0 and wk_kumo_twist_aligned[i] > 0 and 
                tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1] and  # Tenkan crosses above Kijun
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # BEARISH: Price below weekly Kumo + Kumo twist bearish (B > A)
            elif (price_vs_wk_kumo_below > 0 and wk_kumo_twist_aligned[i] < 0 and 
                  tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1] and  # Tenkan crosses below Kijun
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Tenkan crosses below Kijun OR price drops below weekly Kumo
            if (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]) or \
               (close[i] < min(wk_senkou_a_aligned[i], wk_senkou_b_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Tenkan crosses above Kijun OR price rises above weekly Kumo
            if (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]) or \
               (close[i] > max(wk_senkou_a_aligned[i], wk_senkou_b_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals