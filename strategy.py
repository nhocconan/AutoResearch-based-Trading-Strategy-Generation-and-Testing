#!/usr/bin/env python3
# 6h_Camarilla_R3S3_Breakout_1wTrend_Volume
# Hypothesis: Camarilla R3/S3 breakouts with weekly trend filter and volume confirmation work across market regimes.
# Weekly trend (1w EMA13) filters direction: only long when above EMA13, short when below.
# Breakout logic: long when close > R3, short when close < S3. Exit on opposite touch (S3 for longs, R3 for shorts).
# Volume confirmation requires >1.5x 20-period average to avoid false breakouts.
# Target: 15-25 trades/year per symbol.

name = "6h_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "6h"
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

    # Camarilla levels from previous day (using daily OHLC)
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # Calculate daily values for typical price, then get Camarilla levels
    # We'll use rolling window of 24 (for 6h data, 24 periods = 4 days) but need daily OHLC
    # Instead, resample conceptually: we need previous day's high, low, close
    # Since we can't resample, we'll approximate using rolling window of 4 (previous 4*6h = 1 day)
    # But better: use actual daily data via get_htf_data for '1d'
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day: based on previous day's OHLC
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    # We'll shift by 1 to use previous day's values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)

    # Weekly trend filter: 1w EMA13
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        ema13_1w = np.full_like(close, np.nan)
    else:
        ema13_1w = pd.Series(df_1w['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema13_1w)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Warmup period
        # Skip if any required value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema13_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: close > R3, above weekly EMA13, volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema13_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: close < S3, below weekly EMA13, volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema13_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: close < S3 (opposite touch) or weekly trend turns bearish
            if (close[i] < s3_aligned[i] or close[i] < ema13_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: close > R3 (opposite touch) or weekly trend turns bullish
            if (close[i] > r3_aligned[i] or close[i] > ema13_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals