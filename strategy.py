#!/usr/bin/env python3

# 4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
# Hypothesis: On 4h timeframe, enter long when price breaks above Camarilla R1 with volume >2x average and 1d EMA34 trending up; enter short when price breaks below Camarilla S3 with volume >2x average and 1d EMA34 trending down. Uses volume confirmation and trend filter to reduce false signals. Targets 20-50 trades/year to minimize fee drag and improve generalization across bull/bear markets.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
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

    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R1 = close + (high - low) * 1.12/6, S1 = close - (high - low) * 1.12/6, S3 = close - (high - low) * 1.12/4
    range_1d = high_1d - low_1d
    camarilla_r1 = close_1d + range_1d * 1.12 / 6
    camarilla_s1 = close_1d - range_1d * 1.12 / 6
    camarilla_s3 = close_1d - range_1d * 1.12 / 4

    # Use previous 1d bar's levels (shift by 1)
    camarilla_r1_prev = np.roll(camarilla_r1, 1)
    camarilla_s1_prev = np.roll(camarilla_s1, 1)
    camarilla_s3_prev = np.roll(camarilla_s3, 1)
    camarilla_r1_prev[0] = np.nan
    camarilla_s1_prev[0] = np.nan
    camarilla_s3_prev[0] = np.nan

    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_prev)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_prev)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_prev)

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 2x 12-period average (approx 24 hours on 4h)
    vol_avg_12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_avg_12[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Camarilla R1 + 1d uptrend + volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and
                volume[i] > vol_avg_12[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3 + 1d downtrend + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and
                  volume[i] > vol_avg_12[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S1 OR trend turns down
            if close[i] < camarilla_s1_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R1 OR trend turns up
            if close[i] > camarilla_r1_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals