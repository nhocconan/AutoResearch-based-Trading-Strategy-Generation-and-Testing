#!/usr/bin/env python3
"""
4h_Donchian_Breakout_With_1D_Trend_And_Volume_Filter
Hypothesis: Donchian(20) breakouts combined with 1d EMA50 trend filter and 1d volume spike
provide robust entries with low frequency. Designed for 20-50 trades/year to minimize
fee drag while capturing trends in both bull and bear markets. Uses 1d EMA for trend
and volume confirmation to avoid false breakouts in choppy markets.
"""

name = "4h_Donchian_Breakout_With_1D_Trend_And_Volume_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values

    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Calculate 1d volume average (20-period)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)

    # Calculate Donchian channels (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_avg = vol_avg_20_1d_aligned[i]
        vol_1d = volume_1d[i // 96]  # 96 4h bars per day

        if np.isnan(donch_high) or np.isnan(donch_low) or np.isnan(ema_trend) or np.isnan(vol_avg):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian high + above 1d EMA50 + volume spike
            if close[i] > donch_high and close[i] > ema_trend and vol_1d > vol_avg * 2.0:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + below 1d EMA50 + volume spike
            elif close[i] < donch_low and close[i] < ema_trend and vol_1d > vol_avg * 2.0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low
            if close[i] < donch_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high
            if close[i] > donch_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals