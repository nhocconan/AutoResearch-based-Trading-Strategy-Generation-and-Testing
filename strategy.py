#!/usr/bin/env python3
# 4h_1D_VolumeSpike_DonchianBreakout
# Hypothesis: On 4h timeframe, enter long when price breaks above 20-period Donchian high with volume spike (>2x 20-period average) and bullish 1d EMA34 trend; enter short when price breaks below 20-period Donchian low with volume spike and bearish 1d EMA34 trend. Exit on opposite Donchian break or trend reversal. Designed for 20-50 trades/year to minimize fee drag while capturing strong momentum moves in both bull and bear markets.

name = "4h_1D_VolumeSpike_DonchianBreakout"
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

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate daily EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Volume confirmation: current volume > 2.0x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Daily trend filter
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]

        if position == 0:
            # LONG: Price breaks above Donchian high with bullish daily trend and volume spike
            if close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1] and bullish_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low with bearish daily trend and volume spike
            elif close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1] and bearish_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low or daily trend turns bearish
            if close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high or daily trend turns bullish
            if close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals