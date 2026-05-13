#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Price breaking above/below Camarilla R3/S3 levels with 1d EMA34 trend filter and volume confirmation captures momentum with controlled trade frequency.
# Works in bull markets via breakouts above R3 and in bear markets via breakdowns below S3.
# Uses 1d EMA34 to filter trend direction and volume spike for confirmation, reducing false signals.
# Target: 12-37 trades per year per symbol to minimize fee drag on 12h timeframe.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for Camarilla pivot calculation and EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # where C, H, L are from previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    pivot = (prev_high + prev_low + prev_close) / 3
    r3 = pivot + (prev_high - prev_low) * 1.1 / 4
    s3 = pivot - (prev_high - prev_low) * 1.1 / 4
    
    # Align to 12h timeframe (available after previous day close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume filter: >2.0x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
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
            # LONG: Close above R3 + 1d EMA34 uptrend + volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.30
                position = 1
            # SHORT: Close below S3 + 1d EMA34 downtrend + volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below S3 or volatility drop
            if close[i] < s3_aligned[i] or volume[i] < vol_avg_20[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Close above R3 or volatility drop
            if close[i] > r3_aligned[i] or volume[i] < vol_avg_20[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30

    return signals