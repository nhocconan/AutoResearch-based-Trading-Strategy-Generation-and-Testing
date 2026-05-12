#!/usr/bin/env python3
# 1d_Chaikin_Money_Flow_Reverse
# Hypothesis: Chaikin Money Flow (CMF) reversal on daily timeframe with weekly trend filter.
# Long when CMF crosses above +0.1 with weekly uptrend (price > weekly EMA20).
# Short when CMF crosses below -0.1 with weekly downtrend (price < weekly EMA20).
# Exit when CMF crosses back toward zero (long exit < 0.05, short exit > -0.05).
# Designed for low trade frequency (7-25/year) to avoid fee drag. Works in bull/bear markets by following weekly trend direction.

name = "1d_Chaikin_Money_Flow_Reverse"
timeframe = "1d"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Calculate weekly EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)

    # Calculate Chaikin Money Flow (20-period) on daily data
    # MFM = ((close - low) - (high - close)) / (high - low)
    # MFV = MFM * volume
    # CMF = 20-period sum of MFV / 20-period sum of volume
    high_low = high - low
    # Avoid division by zero
    high_low_safe = np.where(high_low == 0, 1e-10, high_low)
    mfm = ((close - low) - (high - close)) / high_low_safe
    mfv = mfm * volume

    # Calculate 20-period sums
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    volume_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    cmf = np.divide(mfv_sum, volume_sum, out=np.zeros_like(mfv_sum), where=volume_sum!=0)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(cmf[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: CMF crosses above +0.1 with weekly uptrend
            if (cmf[i] > 0.1 and cmf[i-1] <= 0.1 and close[i] > ema20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: CMF crosses below -0.1 with weekly downtrend
            elif (cmf[i] < -0.1 and cmf[i-1] >= -0.1 and close[i] < ema20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: CMF crosses back below 0.05
            if cmf[i] < 0.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: CMF crosses back above -0.05
            if cmf[i] > -0.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals