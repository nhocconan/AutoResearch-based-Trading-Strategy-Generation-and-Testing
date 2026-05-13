#!/usr/bin/env python3
# 1h_Donchian_Breakout_4hTrend_Volume
# Hypothesis: Use 4h trend (EMA200) and Donchian channel breakout (20-period) on 1h for entry timing, with volume confirmation.
# Long when price breaks above Donchian upper band with 4h uptrend and volume spike.
# Short when price breaks below Donchian lower band with 4h downtrend and volume spike.
# Exit on opposite Donchian band break or trend reversal.
# Uses 4h for signal direction (trend filter) and 1h for entry timing to reduce noise and control trade frequency.
# Designed for 1h timeframe with target of 15-35 trades/year to avoid fee drag.

name = "1h_Donchian_Breakout_4hTrend_Volume"
timeframe = "1h"
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

    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # 4h trend: EMA200
    close_4h = df_4h['close'].values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # 1h Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: volume > 2.0 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(ema200_4h_aligned[i]) or 
            np.isnan(high_20[i]) or 
            np.isnan(low_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above upper Donchian + 4h uptrend + volume spike
            if close[i] > high_20[i] and close[i] > ema200_4h_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Break below lower Donchian + 4h downtrend + volume spike
            elif close[i] < low_20[i] and close[i] < ema200_4h_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Break below lower Donchian or 4h trend turns down
            if close[i] < low_20[i] or close[i] < ema200_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Break above upper Donchian or 4h trend turns up
            if close[i] > high_20[i] or close[i] > ema200_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals