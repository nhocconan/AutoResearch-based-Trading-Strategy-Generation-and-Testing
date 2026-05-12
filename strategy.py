# 6h_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Weekly trend filter with daily Camarilla R3/S3 breakout on 6h timeframe.
# Uses Camarilla R3/S3 from prior day's OHLC for mean-reversion at extremes.
# In bull markets (price > weekly EMA50): long on R3 breakout, short on S3 breakdown.
# In bear markets (price < weekly EMA50): short on R3 breakdown, long on S3 breakout.
# Volume spike confirmation (2x 20-period SMA) reduces false breakouts.
# Designed for low trade frequency (12-37/year) to avoid fee drag.

name = "6h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "6h"
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

    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla R3/S3 for each day: based on prior day's OHLC
    # R3 = close + 1.1 * (high - low) / 6
    # S3 = close - 1.1 * (high - low) / 6
    rng_1d = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * rng_1d / 6
    camarilla_s3 = close_1d - 1.1 * rng_1d / 6

    # Align Camarilla levels to 6h timeframe (use prior day's levels for current day)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)

    # Get weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Calculate volume spike threshold (2.0x 20-period SMA on 6h)
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Determine trend based on weekly EMA50
        is_uptrend = close[i] > ema50_1w_aligned[i]

        if position == 0:
            # In uptrend: look for R3 breakout (long) or S3 breakdown (short)
            # In downtrend: look for S3 breakout (long) or R3 breakdown (short)
            if is_uptrend:
                # Uptrend: long on R3 breakout, short on S3 breakdown
                if (close[i] > camarilla_r3_aligned[i] and 
                    volume[i] > volume_sma20[i]):
                    signals[i] = 0.25
                    position = 1
                elif (close[i] < camarilla_s3_aligned[i] and 
                      volume[i] > volume_sma20[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Downtrend: long on S3 breakout, short on R3 breakdown
                if (close[i] > camarilla_s3_aligned[i] and 
                    volume[i] > volume_sma20[i]):
                    signals[i] = 0.25
                    position = 1
                elif (close[i] < camarilla_r3_aligned[i] and 
                      volume[i] > volume_sma20[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price touches or crosses below opposite level based on trend
            if is_uptrend:
                # In uptrend, exit long at S3
                if close[i] < camarilla_s3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In downtrend, exit long at R3
                if close[i] < camarilla_r3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price touches or crosses above opposite level based on trend
            if is_uptrend:
                # In uptrend, exit short at R3
                if close[i] > camarilla_r3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In downtrend, exit short at S3
                if close[i] > camarilla_s3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25

    return signals