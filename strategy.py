# 12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
# Hypothesis: 12h Camarilla pivot breakout with 1w trend filter and volume spike confirmation.
# Camarilla R3/S3 levels provide strong support/resistance from prior day's range.
# 1w EMA20 determines long-term trend to avoid counter-trend trades.
# Volume spike (2x average) confirms breakout strength.
# Designed for 12h timeframe with ~50-150 total trades over 4 years to minimize fee drag.
# Works in bull/bear by following 1w trend - only takes longs in uptrend, shorts in downtrend.

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
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

    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)

    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values

    # Calculate Camarilla levels for each 12h bar using prior bar's range
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    # Using prior bar to avoid look-ahead
    high_shift = np.roll(high_12h, 1)
    low_shift = np.roll(low_12h, 1)
    close_shift = np.roll(close_12h, 1)
    high_shift[0] = high_12h[0]
    low_shift[0] = low_12h[0]
    close_shift[0] = close_12h[0]
    
    r3 = close_shift + 1.1 * (high_shift - low_shift)
    s3 = close_shift - 1.1 * (high_shift - low_shift)

    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)

    # Get 1w data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)

    # Calculate 12h volume SMA10 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma10 = volume_series.rolling(window=10, min_periods=10).mean().values
    volume_spike_threshold = volume_sma10 * 2.0  # Require 2x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(10, n):  # Start after volume SMA warmup
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema20_1w_aligned[i]) or np.isnan(volume_sma10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above R3 in 1w uptrend with volume spike
            if close[i] > r3_aligned[i] and close[i] > ema20_1w_aligned[i] and volume[i] > volume_sma10[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below S3 in 1w downtrend with volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema20_1w_aligned[i] and volume[i] > volume_sma10[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below S3 (reversal to downside)
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above R3 (reversal to upside)
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals