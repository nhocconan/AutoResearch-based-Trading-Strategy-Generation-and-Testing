#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Breakouts at Camarilla R3/S3 levels with 1-week EMA trend filter and volume spike captures strong momentum moves while minimizing false signals. The weekly trend filter ensures alignment with higher timeframe momentum, reducing whipsaws in ranging markets and improving performance in both bull and bear cycles.
# Uses Camarilla pivot levels from prior day, weekly EMA for trend, and volume confirmation for entry. Exits on mean reversion to prior day's close (pivot base) to avoid overstaying.
# Target: 20-40 trades/year on 4h to stay within optimal range while capturing significant moves.

name = "4h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
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

    # Get prior day's OHLC for Camarilla calculation
    # Use 1d data from mtf_data
    df_1d = get_htf_data(prices, '1d')
    # Prior day's close, high, low
    prev_close = df_1d['close'].shift(1).values  # Shifted by 1 to get prior day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values

    # Calculate Camarilla levels for prior day
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # We use R3 and S3 for entries
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range
    s3 = prev_close - 1.1 * camarilla_range
    pivot = (prev_high + prev_low + prev_close) / 3.0  # Pivot point (not used directly but for context)

    # Align Camarilla levels to 4h timeframe (already aligned by shift, but ensure no look-ahead)
    # Since we used shift(1) on 1d data, it's already prior day's values, safe to use
    # But we still use align_htf_to_ltf for consistency and to handle any edge cases
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)

    # Get 1-week EMA for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema10_1w = pd.Series(df_1w['close']).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)

    # Volume confirmation: volume > 2.0x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema10_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above R3 + 1w EMA uptrend + volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema10_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Close below S3 + 1w EMA downtrend + volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema10_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Mean reversion to prior day's close (pivot base)
            if close[i] < df_1d['close'].shift(1).values[i]:  # Prior day's close
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Mean reversion to prior day's close (pivot base)
            if close[i] > df_1d['close'].shift(1).values[i]:  # Prior day's close
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals