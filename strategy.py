#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume
# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and 1d volume spike confirmation.
# Uses 4h for trend direction and 1d for volume confirmation to avoid counter-trend trades.
# Designed for 60-150 total trades over 4 years (15-37/year) to minimize fee drift.
# Works in bull/bear by following 4h trend and requiring volume confirmation.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
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

    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)

    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)

    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    volume_1d = df_1d['volume'].values
    volume_sma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma20_1d)

    # Calculate Camarilla levels for 1h: based on previous day's OHLC
    # We'll use the 1d OHLC from the previous completed day
    # Since we already have df_1d, we can compute Camarilla levels from it
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C, H, L are from previous day
    # We need to shift the 1d data by 1 to get previous day's values
    close_1d_prev = np.roll(volume_1d, 1)  # temporary, will replace
    high_1d_prev = np.roll(df_1d['high'].values, 1)
    low_1d_prev = np.roll(df_1d['low'].values, 1)
    close_1d_prev = np.roll(df_1d['close'].values, 1)

    # First value is invalid due to roll
    close_1d_prev[0] = np.nan
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan

    # Calculate Camarilla R1 and S1 from previous day
    camarilla_r1 = close_1d_prev + (high_1d_prev - low_1d_prev) * 1.1 / 12
    camarilla_s1 = close_1d_prev - (high_1d_prev - low_1d_prev) * 1.1 / 12

    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(1, n):  # start from 1 to avoid issues with rolled values
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema50_4h_aligned[i]) or np.isnan(volume_sma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above R1 in 4h uptrend with 1d volume spike
            if close[i] > camarilla_r1_aligned[i] and close[i] > ema50_4h_aligned[i] and volume[i] > volume_sma20_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Break below S1 in 4h downtrend with 1d volume spike
            elif close[i] < camarilla_s1_aligned[i] and close[i] < ema50_4h_aligned[i] and volume[i] > volume_sma20_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below S1 (reversal signal)
            if close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price closes above R1 (reversal signal)
            if close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals