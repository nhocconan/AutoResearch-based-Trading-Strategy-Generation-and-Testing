#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Price breaking out of Camarilla R3/S3 levels on 12h with 1d trend and volume confirmation captures strong momentum moves while avoiding false breakouts.
# Uses daily EMA34 trend filter and volume spike confirmation to filter noise.
# Entry: Long when close > R3 + daily EMA34 uptrend + volume spike; Short when close < S3 + daily EMA34 downtrend + volume spike.
# Exit: Mean reversion to daily EMA34 to avoid overstaying in extended moves.
# Target: 12-30 trades/year on 12h to stay within optimal range.

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

    # Get daily data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    daily_close = df_1d['close'].values
    ema34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate previous day's Camarilla levels (R3, S3)
    # Camarilla formula: R3 = close + (high - low) * 1.1/2, S3 = close - (high - low) * 1.1/2
    prev_daily_close = df_1d['close'].shift(1).values
    prev_daily_high = df_1d['high'].shift(1).values
    prev_daily_low = df_1d['low'].shift(1).values
    camarilla_r3 = prev_daily_close + (prev_daily_high - prev_daily_low) * 1.1 / 2
    camarilla_s3 = prev_daily_close - (prev_daily_high - prev_daily_low) * 1.1 / 2
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)

    # Volume confirmation: volume > 1.8x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):  # Start after EMA34 warmup
        # Skip if any required value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above R3 + daily EMA34 uptrend + volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Close below S3 + daily EMA34 downtrend + volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Mean reversion to daily EMA34
            if close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Mean reversion to daily EMA34
            if close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals