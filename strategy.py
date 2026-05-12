#!/usr/bin/env python3
# 12h_Ichimoku_Kumo_Twist_Trend_1d
# Hypothesis: Use Ichimoku Cloud on 12h to identify trend direction (price above/below Kumo) and Kumo twist (Senkou Span A/B crossover) for momentum confirmation. Filter with 1d EMA50 trend and volume spike (>1.5x 20-period average) to avoid false signals. Target 20-40 trades/year to minimize fee drift and work in both bull/bear markets via multi-timeframe trend alignment.

name = "12h_Ichimoku_Kumo_Twist_Trend_1d"
timeframe = "12h"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Ichimoku Cloud (9, 26, 52) on 12h
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2

    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2

    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + low)/2, shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)

    # Align Ichimoku components to 12h timeframe (no shift needed as calculation uses historical data)
    # Kumo twist: Senkou A crossing above/below Senkou B
    # We use current Senkou A and B to determine cloud thickness and twist
    # For trend: price above/below cloud
    # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # start after Senkou B calculation window
        # Skip if any required value is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price above cloud + Kumo bullish twist (Senkou A > Senkou B) + price > 1d EMA50 + volume spike
            if (close[i] > cloud_top[i] and 
                senkou_a[i] > senkou_b[i] and
                close[i] > ema50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price below cloud + Kumo bearish twist (Senkou A < Senkou B) + price < 1d EMA50 + volume spike
            elif (close[i] < cloud_bottom[i] and 
                  senkou_a[i] < senkou_b[i] and
                  close[i] < ema50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls below cloud OR Kumo turns bearish
            if close[i] < cloud_top[i] or senkou_a[i] < senkou_b[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises above cloud OR Kumo turns bullish
            if close[i] > cloud_bottom[i] or senkou_a[i] > senkou_b[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals