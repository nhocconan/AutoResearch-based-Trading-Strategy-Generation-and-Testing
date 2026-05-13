#!/usr/bin/env python3
# 12h_Camarilla_R3S3_Breakout_1dTrend_VolumeS
# Hypothesis: Enter long when price breaks above Camarilla R3 level on 12h timeframe with 1d uptrend filter and volume confirmation.
# Enter short when price breaks below Camarilla S3 level on 12h timeframe with 1d downtrend filter and volume confirmation.
# Camarilla levels provide statistically significant support/resistance from prior day's range.
# Trend filter (1d EMA34) ensures alignment with higher timeframe momentum, reducing false breakouts in choppy markets.
# Volume confirmation (volume > 1.5 * 20-period average) ensures institutional participation.
# Designed for low frequency (~20-40 trades/year) to minimize fee drag and work in both bull and bear markets.

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeS"
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

    # Get daily data for Camarilla calculation and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from prior day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # Using prior day's data to avoid look-ahead
    range_1d = high_1d - low_1d
    pp = (high_1d + low_1d + close_1d) / 3.0
    r3 = pp + (range_1d * 1.1 / 2.0)  # R3 = PP + 1.1*(H-L)/2
    s3 = pp - (range_1d * 1.1 / 2.0)  # S3 = PP - 1.1*(H-L)/2
    
    # Daily trend: EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily indicators to 12h timeframe (2 bars per day)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average (~10 days at 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * vol_ma_20
    
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
            # LONG: Close > R3 + daily uptrend + volume confirmation
            if close[i] > r3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < S3 + daily downtrend + volume confirmation
            elif close[i] < s3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below EMA34 or below S3 (stop and reverse)
            if close[i] < ema34_1d_aligned[i] or close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above EMA34 or above R3 (stop and reverse)
            if close[i] > ema34_1d_aligned[i] or close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals