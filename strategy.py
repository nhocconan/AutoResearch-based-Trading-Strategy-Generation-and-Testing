#!/usr/bin/env python3
# 12h_DonchianBreakout_Volume_1dTrend
# Hypothesis: Trade Donchian(20) breakouts on 12h with volume confirmation and 1d EMA trend filter.
# Works in bull/bear by following the 1d trend. Volume ensures breakout strength.
# Low turnover expected (~20-40/year) to avoid fee drag.

name = "12h_DonchianBreakout_Volume_1dTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    # Calculate Donchian channels (20-period) on 12h
    high_max = np.full(n, np.nan)
    low_min = np.full(n, np.nan)
    for i in range(20, n):
        high_max[i] = np.max(high[i-20:i])
        low_min[i] = np.min(low[i-20:i])

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    # Get 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(volume_spike[i]) or np.isnan(ema_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above Donchian high with volume spike and 1d EMA50 uptrend
            if close[i] > high_max[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below Donchian low with volume spike and 1d EMA50 downtrend
            elif close[i] < low_min[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls back below Donchian low (failed breakout or reversal)
            if close[i] < low_min[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises back above Donchian high (failed breakout or reversal)
            if close[i] > high_max[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals