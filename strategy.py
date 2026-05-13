#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla R3/S3 breakouts on 12h timeframe with 1d trend filter and volume confirmation capture institutional breakout moves while avoiding whipsaws. Works in bull markets via momentum continuation and in bear markets via mean-reversion failure at extreme levels.
# Uses 1d EMA34 for trend filter to ensure alignment with higher timeframe momentum.
# Entry: Long when price breaks above R3 + 1d EMA34 uptrend + volume spike; Short when price breaks below S3 + 1d EMA34 downtrend + volume spike.
# Exit: Mean reversion to R4/S4 levels or opposite Camarilla level to avoid overstaying extended moves.
# Target: 20-35 trades/year on 12h to stay within optimal range while capturing significant moves.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
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

    # Calculate 1d Camarilla levels (using previous day's OHLC)
    # Get daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day: based on previous day's OHLC
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.1 * (High - Low)
    # S3 = Close - 1.1 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    R3_1d = prev_close + 1.1 * (prev_high - prev_low)
    S3_1d = prev_close - 1.1 * (prev_high - prev_low)
    R4_1d = prev_close + 1.5 * (prev_high - prev_low)
    S4_1d = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align to 12h timeframe
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    R4_1d_aligned = align_htf_to_ltf(prices, df_1d, R4_1d)
    S4_1d_aligned = align_htf_to_ltf(prices, df_1d, S4_1d)

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]) or 
            np.isnan(R4_1d_aligned[i]) or np.isnan(S4_1d_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above R3 + 1d EMA34 uptrend + volume spike
            if (close[i] > R3_1d_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Close below S3 + 1d EMA34 downtrend + volume spike
            elif (close[i] < S3_1d_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Mean reversion to R4 level
            if close[i] < R4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Mean reversion to S4 level
            if close[i] > S4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals