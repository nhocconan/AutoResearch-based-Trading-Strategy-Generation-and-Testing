#!/usr/bin/env python3
# 4h_Trix_Trend_With_Volume_Confirmation
# Strategy uses TRIX (12-period EMA applied 3 times) for momentum, confirmed by 12h EMA trend and volume spike.
# Long when TRIX crosses above zero in uptrend with volume spike; short when TRIX crosses below zero in downtrend with volume spike.
# Exit when TRIX crosses back through zero or trend reverses.
# Designed for low-frequency, high-conviction trades with strong trend alignment to minimize whipsaw and fee drag.
# Works in bull (TRIX > 0 in uptrend) and bear (TRIX < 0 in downtrend).

name = "4h_Trix_Trend_With_Volume_Confirmation"
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

    # Get 12h data for TRIX and trend
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values

    # Calculate TRIX: EMA(EMA(EMA(close, 12), 12), 12)
    ema1 = pd.Series(close_12h).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = (ema3 / ema3.shift(1) - 1) * 100  # Percentage change
    trix = trix.values

    # Zero line
    zero_line = np.zeros_like(trix)

    # 12h EMA34 for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values

    # Volume spike: volume > 2.0 * 6-period average (1 day at 4h)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_spike = volume > 2.0 * vol_ma_6

    # Align 12h indicators to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_12h, trix)
    zero_line_aligned = align_htf_to_ltf(prices, df_12h, zero_line)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(trix_aligned[i]) or 
            np.isnan(zero_line_aligned[i]) or 
            np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + uptrend + volume spike
            if trix_aligned[i] > zero_line_aligned[i] and trix_aligned[i-1] <= zero_line_aligned[i-1] and ema34_12h_aligned[i] > close[i] and volume_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + downtrend + volume spike
            elif trix_aligned[i] < zero_line_aligned[i] and trix_aligned[i-1] >= zero_line_aligned[i-1] and ema34_12h_aligned[i] < close[i] and volume_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero OR trend reversal
            if trix_aligned[i] < zero_line_aligned[i] and trix_aligned[i-1] >= zero_line_aligned[i-1] or ema34_12h_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero OR trend reversal
            if trix_aligned[i] > zero_line_aligned[i] and trix_aligned[i-1] <= zero_line_aligned[i-1] or ema34_12h_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals