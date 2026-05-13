#!/usr/bin/env python3
# 12h_Donchian20_Breakout_1dTrend_Volume
# Hypothesis: 12h Donchian channel breakout with 1d EMA200 trend filter and volume confirmation.
# Long when price breaks above Donchian upper band in uptrend with volume spike (>1.5x 20-period avg).
# Short when price breaks below Donchian lower band in downtrend with volume spike.
# Exit when price returns to Donchian middle line (mean of upper/lower) or trend changes.
# Designed for low trade frequency (50-150 total over 4 years) with clear trend-following logic.

name = "12h_Donchian20_Breakout_1dTrend_Volume"
timeframe = "12h"
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

    # Get 12h data for Donchian channel (20-period)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Donchian channel: upper = max(high,20), lower = min(low,20), middle = (upper+lower)/2
    high_series = pd.Series(df_12h['high'])
    low_series = pd.Series(df_12h['low'])
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Align Donchian levels to lower timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_12h, donchian_middle)

    # Get daily data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200 for trend filter
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)

    # Volume filter: >1.5x 20-period average (using lower timeframe volume)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian upper + price above 1d EMA200 (uptrend) + volume spike
            if (close[i] > donchian_upper_aligned[i] and 
                close[i] > ema_200_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower + price below 1d EMA200 (downtrend) + volume spike
            elif (close[i] < donchian_lower_aligned[i] and 
                  close[i] < ema_200_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to Donchian middle or trend changes (price below EMA200)
            if (close[i] <= donchian_middle_aligned[i] or close[i] < ema_200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to Donchian middle or trend changes (price above EMA200)
            if (close[i] >= donchian_middle_aligned[i] or close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals