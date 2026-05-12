#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout
Hypothesis: Use Ichimoku Cloud on daily timeframe to define trend and support/resistance, with 6h Tenkan-Kijun cross for entry timing. In bull markets, price above cloud with bullish TK cross signals long; in bear markets, price below cloud with bearish TK cross signals short. Volume confirmation filters false breaks. Works in both regimes by aligning with higher timeframe Ichimoku structure.
"""

name = "6h_Ichimoku_Cloud_Breakout"
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

    # Get daily data for Ichimoku (call once before loop)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 52:
        return np.zeros(n)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values

    # Ichimoku components on daily
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_daily).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_daily).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2

    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high_daily).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_daily).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2

    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_daily).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_daily).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_senkou_b + min_low_senkou_b) / 2)

    # Align Ichimoku components to 6h timeframe (wait for daily close)
    tenkan_aligned = align_htf_to_ltf(prices, df_daily, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_daily, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_daily, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_daily, senkou_b)

    # Tenkan-Kijun cross on 6h for entry timing
    period_tenkan_6h = 9
    period_kijun_6h = 26
    max_high_tenkan_6h = pd.Series(high).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).max().values
    min_low_tenkan_6h = pd.Series(low).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).min().values
    tenkan_6h = (max_high_tenkan_6h + min_low_tenkan_6h) / 2
    max_high_kijun_6h = pd.Series(high).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).max().values
    min_low_kijun_6h = pd.Series(low).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).min().values
    kijun_6h = (max_high_kijun_6h + min_low_kijun_6h) / 2

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any Ichimoku values are not yet available
        if np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        tenkan_val = tenkan_6h[i]
        kijun_val = kijun_6h[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        vol_avg_val = vol_avg_20[i]

        if np.isnan(tenkan_val) or np.isnan(kijun_val) or np.isnan(senkou_a_val) or np.isnan(senkou_b_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Determine cloud boundaries and price position
        upper_cloud = max(senkou_a_val, senkou_b_val)
        lower_cloud = min(senkou_a_val, senkou_b_val)
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud

        if position == 0:
            # LONG: price above cloud + bullish TK cross + volume confirmation
            if price_above_cloud and tenkan_val > kijun_val and tenkan_6h[i-1] <= kijun_6h[i-1] and volume[i] > vol_avg_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: price below cloud + bearish TK cross + volume confirmation
            elif price_below_cloud and tenkan_val < kijun_val and tenkan_6h[i-1] >= kijun_6h[i-1] and volume[i] > vol_avg_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls below cloud or bearish TK cross
            if price_below_cloud or (tenkan_val < kijun_val and tenkan_6h[i-1] >= kijun_6h[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises above cloud or bullish TK cross
            if price_above_cloud or (tenkan_val > kijun_val and tenkan_6h[i-1] <= kijun_6h[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals