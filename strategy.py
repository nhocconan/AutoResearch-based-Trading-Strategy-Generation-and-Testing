#!/usr/bin/env python3
# 4h_Ichimoku_Cloud_Breakout_Trend_Volume
# Hypothesis: Ichimoku Cloud breakouts with 1d trend filter and volume confirmation capture momentum in both bull and bear markets.
# Uses the Kumo (cloud) for dynamic support/resistance and Tenkan/Kijun crossovers for entry confirmation.
# Target: 25-35 trades/year per symbol.

name = "4h_Ichimoku_Cloud_Breakout_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    # 1d EMA50 for trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Ichimoku Cloud (9, 26, 52)
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

    # The actual cloud (Kumo) boundaries - shifted forward by 26 periods
    # So we need to shift the calculated Senkou spans back to align with current price
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    # For the first 26 periods, we don't have cloud data
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan

    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    cloud_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # Start after Ichimoku calculations are valid
        if np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above cloud + 1d uptrend + volume spike
            if close[i] > cloud_top[i] and close[i] > ema50_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below cloud + 1d downtrend + volume spike
            elif close[i] < cloud_bottom[i] and close[i] < ema50_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below cloud or 1d trend turns down
            if close[i] < cloud_bottom[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above cloud or 1d trend turns up
            if close[i] > cloud_top[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals