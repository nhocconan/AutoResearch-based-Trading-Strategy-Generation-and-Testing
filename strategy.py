#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Use 4-hour Camarilla pivot level breakouts with 1-day EMA trend filter and volume confirmation.
# Camarilla levels provide high-probability support/resistance; EMA filter ensures alignment with higher timeframe trend.
# Works in bull markets by buying breakouts above R3 in uptrends and in bear markets by selling breakdowns below S3 in downtrends.
# Volume confirmation filters out false breakouts. Target: 20-50 trades per year.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla levels require previous day's data, so we shift by 1
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low

    # Calculate Camarilla levels
    R3 = prev_close + 1.1 * prev_range * 1.1 / 2  # Previous close + 1.1 * range * 1.1/2
    S3 = prev_close - 1.1 * prev_range * 1.1 / 2  # Previous close - 1.1 * range * 1.1/2

    # Align Camarilla levels to 4h timeframe (wait for previous day to complete)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg_20[i]) or 
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above R3 + price above 1d EMA (bullish trend) + volume spike
            if (close[i] > R3_aligned[i] and 
                close[i] > ema_34_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S3 + price below 1d EMA (bearish trend) + volume spike
            elif (close[i] < S3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below S3 or price below 1d EMA
            if (close[i] < S3_aligned[i] or close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above R3 or price above 1d EMA
            if (close[i] > R3_aligned[i] or close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals