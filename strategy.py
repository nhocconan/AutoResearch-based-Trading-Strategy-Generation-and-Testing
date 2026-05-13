#!/usr/bin/env python3
# 4h_Donchian_Breakout_1dTrend_Volume_Spike
# Hypothesis: 4h Donchian channel breakout with 1d trend filter and volume spike confirmation.
# Donchian upper/lower bands calculated from 20-period high/low. Long when price breaks above upper band in uptrend (price > 1d EMA50) with volume spike (>2x 20-period average volume). Short when price breaks below lower band in downtrend (price < 1d EMA50) with volume spike.
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (<50/year) to avoid fee drag while capturing strong trending moves.
# Works in both bull and bear markets: trend filter ensures alignment with higher timeframe direction, volume confirms conviction, Donchian breakout captures momentum.

name = "4h_Donchian_Breakout_1dTrend_Volume_Spike"
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

    # Trend filter: 1d EMA50
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Donchian channel (20-period) on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above Donchian upper band in uptrend with volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Donchian lower band in downtrend with volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian lower band or trend turns down
            if close[i] < donchian_low[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian upper band or trend turns up
            if close[i] > donchian_high[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals