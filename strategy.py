#!/usr/bin/env python3
# 4h_Donchian_Breakout_Volume_12hTrend
# Hypothesis: Donchian channel breakouts capture momentum bursts; volume confirms institutional interest; 12h EMA50 trend filter avoids counter-trend trades. Works in bull markets via breakouts and in bear markets via breakdowns. Designed for 4h timeframe to limit trades and avoid overtrading.

name = "4h_Donchian_Breakout_Volume_12hTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)

    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)

    # Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from 12h EMA50
        price_above_12h_ema = close[i] > ema_50_12h_aligned[i]
        price_below_12h_ema = close[i] < ema_50_12h_aligned[i]

        if position == 0:
            # LONG: Price breaks above Donchian upper band with volume and uptrend
            if close[i] > highest_high[i] and volume_ok[i] and price_above_12h_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower band with volume and downtrend
            elif close[i] < lowest_low[i] and volume_ok[i] and price_below_12h_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian lower band or trend turns down
            if close[i] < lowest_low[i] or not price_above_12h_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian upper band or trend turns up
            if close[i] > highest_high[i] or not price_below_12h_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals