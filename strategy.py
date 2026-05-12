#!/usr/bin/env python3

# 6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_HT
# Hypothesis: On 6h timeframe, enter long when price breaks above Camarilla R3 with volume >2x average and 1d EMA34 trending up; enter short when price breaks below Camarilla S3 with volume >2x average and 1d EMA34 trending down.
# Uses weekly trend filter (price above/below weekly EMA50) to avoid counter-trend trades. Targets 12-37 trades/year to minimize fee drag and improve generalization across bull/bear markets.

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_HT"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R3 = close + (high - low) * 1.12/4, S3 = close - (high - low) * 1.12/4
    range_1d = high_1d - low_1d
    camarilla_r3 = close_1d + range_1d * 1.12 / 4
    camarilla_s3 = close_1d - range_1d * 1.12 / 4

    # Use previous 1d bar's levels (shift by 1)
    camarilla_r3_prev = np.roll(camarilla_r3, 1)
    camarilla_s3_prev = np.roll(camarilla_s3, 1)
    camarilla_r3_prev[0] = np.nan
    camarilla_s3_prev[0] = np.nan

    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_prev)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_prev)

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Volume confirmation: volume > 2x 24-period average (approx 12 hours)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(80, n):
        # Skip if any required value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Camarilla R3 + 1d uptrend + weekly uptrend + volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and
                volume[i] > vol_avg_24[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3 + 1d downtrend + weekly downtrend + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and
                  volume[i] > vol_avg_24[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S3 OR trend turns down
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R3 OR trend turns up
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals