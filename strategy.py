#!/usr/bin/env python3
# 4h_Donchian20_Breakout_1dEMA200_Trend_Volume
# Hypothesis: 4h Donchian(20) breakout with 1d EMA200 trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel in uptrend (price > 1d EMA200) with volume spike.
# Short when price breaks below lower Donchian channel in downtrend (price < 1d EMA200) with volume spike.
# Exit when price returns to the middle of the Donchian channel.
# Designed for low trade frequency (~25-50 trades/year) to avoid fee drag and work in both bull and bear markets.

name = "4h_Donchian20_Breakout_1dEMA200_Trend_Volume"
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

    # Get 4h data for Donchian channel calculation
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h Donchian(20) channels: upper, lower, and middle
    high_20 = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    upper_20 = high_20
    lower_20 = low_20
    middle_20 = (upper_20 + lower_20) / 2
    
    # Align 4h Donchian levels to 4h timeframe (no shift needed as we use current bar's values)
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    middle_20_aligned = align_htf_to_ltf(prices, df_4h, middle_20)

    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200 for trend filter
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(middle_20_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above upper Donchian + price above 1d EMA200 (uptrend) + volume spike
            if (close[i] > upper_20_aligned[i] and 
                close[i] > ema_200_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian + price below 1d EMA200 (downtrend) + volume spike
            elif (close[i] < lower_20_aligned[i] and 
                  close[i] < ema_200_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to middle of Donchian channel
            if close[i] <= middle_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to middle of Donchian channel
            if close[i] >= middle_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals