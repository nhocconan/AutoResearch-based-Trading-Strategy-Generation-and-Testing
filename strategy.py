#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend
# Hypothesis: Enter long when price breaks above Camarilla R3 level with 1d uptrend and volume spike; enter short when price breaks below S3 level with 1d downtrend and volume spike.
# Camarilla levels provide precise support/resistance based on prior day's range. Trend filter ensures alignment with higher timeframe momentum.
# Works in bull (breakouts above R3 in uptrend) and bear (breakdowns below S3 in downtrend).
# Low frequency due to specific level breaks and volume confirmation requirement.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend"
timeframe = "12h"
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

    # Get daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Camarilla levels (R3, S3) from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day (based on previous day's range)
    # R3 = close + 1.1 * (high - low) / 6
    # S3 = close - 1.1 * (high - low) / 6
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 6
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 6
    
    # Daily trend: EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily indicators to 12h timeframe (2 bars per day)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: volume > 2.0 * 2-period average (1 day worth at 12h)
    vol_ma_2 = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    volume_spike = volume > 2.0 * vol_ma_2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > R3 + daily uptrend + volume spike
            if close[i] > r3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < S3 + daily downtrend + volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below EMA34
            if close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above EMA34
            if close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals