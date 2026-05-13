#!/usr/bin/env python3
# 6h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
# Hypothesis: Breakouts at Camarilla R3/S3 levels (from daily) in the direction of weekly EMA50 trend, confirmed by volume spike.
# Camarilla levels identify key intraday support/resistance; breaks indicate momentum. Weekly trend filter ensures alignment with higher timeframe momentum, reducing false breakouts.
# Works in bull (breakouts above R3 in uptrend) and bear (breakdowns below S3 in downtrend).
# Volume surge confirms institutional participation. Low frequency due to strict level breaks and trend/volume filters.

name = "6h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "6h"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels from previous day
    # R4 = C + ((H-L) * 1.500), R3 = C + ((H-L) * 1.250), etc.
    # We use previous day's HLC to calculate today's levels
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan

    # Camarilla levels
    R3 = prev_close + (prev_high - prev_low) * 1.250
    S3 = prev_close - (prev_high - prev_low) * 1.250
    R4 = prev_close + (prev_high - prev_low) * 1.500
    S4 = prev_close - (prev_high - prev_low) * 1.500

    # Align daily Camarilla levels to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)

    # Volume spike: volume > 2.0 * 4-period average (1 day worth at 6h)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > 2.0 * vol_ma_4

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(R4_aligned[i]) or 
            np.isnan(S4_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > R3 + weekly uptrend + volume spike
            if close[i] > R3_aligned[i] and close[i] > ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < S3 + weekly downtrend + volume spike
            elif close[i] < S3_aligned[i] and close[i] < ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below S3 or trend reversal
            if close[i] < S3_aligned[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above R3 or trend reversal
            if close[i] > R3_aligned[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals