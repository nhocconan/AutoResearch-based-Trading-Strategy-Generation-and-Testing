#!/usr/bin/env python3
# 4h_RVI_Signal_1dTrend_Volume
# Hypothesis: Relative Vigor Index (RVI) crossovers on 4h with 1d EMA trend filter and volume spike confirmation.
# RVI measures conviction of price moves by comparing closing-open ranges to high-low ranges.
# Long when RVI crosses above its signal line in uptrend with volume spike.
# Short when RVI crosses below its signal line in downtrend with volume spike.
# Uses 1d EMA50 for trend filter to avoid counter-trend trades.
# Target: 20-50 trades/year per symbol to minimize fee decay while capturing momentum shifts.

name = "4h_RVI_Signal_1dTrend_Volume"
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
    open_ = prices['open'].values
    volume = prices['volume'].values

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')

    # Calculate RVI (Relative Vigor Index) on 4h
    # Numerator: (close - open) + 2*(close_prev - open_prev) + 2*(close_prev2 - open_prev2) + (close_prev3 - open_prev3)
    # Denominator: (high - low) + 2*(high_prev - low_prev) + 2*(high_prev2 - low_prev2) + (high_prev3 - low_prev3)
    # RVI = SMA(numerator, 4) / SMA(denominator, 4)
    # Signal line = EMA(RVI, 4)

    a = close - open_
    b = close - open_
    c = close - open_
    d = close - open_

    # Shifted values for numerator
    a1 = np.roll(a, 1)
    a2 = np.roll(a, 2)
    a3 = np.roll(a, 3)
    b1 = np.roll(b, 1)
    b2 = np.roll(b, 2)
    b3 = np.roll(b, 3)
    c1 = np.roll(c, 1)
    c2 = np.roll(c, 2)
    c3 = np.roll(c, 3)
    d1 = np.roll(d, 1)
    d2 = np.roll(d, 2)
    d3 = np.roll(d, 3)

    num = (a + 2*b1 + 2*b2 + b3)  # close-open components
    den = (high - low) + 2*(np.roll(high-low, 1)) + 2*(np.roll(high-low, 2)) + (np.roll(high-low, 3))

    # Avoid division by zero
    den = np.where(den == 0, 1e-10, den)
    rvi_raw = num / den

    # Smooth with 4-period SMA
    rvi = pd.Series(rvi_raw).rolling(window=4, min_periods=4).mean().values
    # Signal line: 4-period EMA of RVI
    rvi_signal = pd.Series(rvi).ewm(span=4, adjust=False, min_periods=4).mean().values

    # Trend filter: 1d EMA50
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(rvi[i]) or 
            np.isnan(rvi_signal[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RVI crosses above signal line in uptrend with volume spike
            if (rvi[i] > rvi_signal[i] and 
                rvi[i-1] <= rvi_signal[i-1] and 
                close[i] > ema50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: RVI crosses below signal line in downtrend with volume spike
            elif (rvi[i] < rvi_signal[i] and 
                  rvi[i-1] >= rvi_signal[i-1] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RVI crosses below signal line
            if rvi[i] < rvi_signal[i] and rvi[i-1] >= rvi_signal[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RVI crosses above signal line
            if rvi[i] > rvi_signal[i] and rvi[i-1] <= rvi_signal[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals