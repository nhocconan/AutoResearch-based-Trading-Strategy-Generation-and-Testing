#!/usr/bin/env python3
# 1D_Camarilla_R3_S3_Breakout_WeeklyTrend_Volume
# Hypothesis: Camarilla R3/S3 breakout on daily timeframe with weekly trend filter and volume confirmation.
# This strategy aims to capture momentum breakouts in trending markets while avoiding false breakouts in ranging conditions.
# Works in both bull and bear markets by only trading in the direction of the weekly trend.

name = "1D_Camarilla_R3_S3_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')

    # Calculate Camarilla levels from previous day
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    r3 = close + (high - low) * 1.1 / 2
    s3 = close - (high - low) * 1.1 / 2
    # Shift to get previous day's levels
    r3_prev = np.roll(r3, 1)
    s3_prev = np.roll(s3, 1)
    r3_prev[0] = np.nan
    s3_prev[0] = np.nan

    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(r3_prev[i]) or 
            np.isnan(s3_prev[i]) or 
            np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R3 in uptrend with volume
            if (close[i] > r3_prev[i] and 
                close[i] > ema34_1w_aligned[i] and 
                volume_confirmed[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 in downtrend with volume
            elif (close[i] < s3_prev[i] and 
                  close[i] < ema34_1w_aligned[i] and 
                  volume_confirmed[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 or trend turns down
            if close[i] < s3_prev[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 or trend turns up
            if close[i] > r3_prev[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals