#!/usr/bin/env python3

# 4h_1D_Donchian20_Breakout_VolumeTrend
# Hypothesis: Donchian(20) breakout with 1d EMA trend filter and volume confirmation.
# In bull markets, buy breakouts above 20-period high with uptrend; in bear markets,
# sell breakdowns below 20-period low with downtrend. Volume > 1.5x average confirms
# breakout strength. Exit on opposite Donchian touch or trend reversal. Targets 20-40 trades/year.

name = "4h_1D_Donchian20_Breakout_VolumeTrend"
timeframe = "4h"
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
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Calculate 4h Donchian channels (20-period)
    period = 20
    donchian_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_low = pd.Series(low).rolling(window=period, min_periods=period).min().values

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: price above/below 34-period EMA on 1d
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]

        if position == 0:
            # LONG: Price breaks above Donchian high with bullish trend and volume confirmation
            if close[i] > donchian_high[i] and bullish_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low with bearish trend and volume confirmation
            elif close[i] < donchian_low[i] and bearish_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches Donchian low or trend turns bearish
            if close[i] < donchian_low[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price touches Donchian high or trend turns bullish
            if close[i] > donchian_high[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals