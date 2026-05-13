#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend
# Hypothesis: Enter long when price breaks above Camarilla R3 level with 1d uptrend and volume confirmation; short when breaks below S3 with 1d downtrend and volume confirmation.
# Exit when price crosses back through the prior day's close (Camarilla base level).
# Camarilla levels act as intraday support/resistance; breaks indicate institutional flow.
# Trend filter ensures alignment with higher timeframe momentum, reducing false breakouts.
# Works in bull (breaks above R3 in uptrend) and bear (breaks below S3 in downtrend).
# Low frequency due to strict level breaks and volume confirmation.

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

    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Previous day's OHLC (needed for Camarilla)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: H-L = range
    rng = prev_high - prev_low
    
    # Resistance levels
    r3 = prev_close + rng * 1.1 / 2
    r4 = prev_close + rng * 1.1
    
    # Support levels
    s3 = prev_close - rng * 1.1 / 2
    s4 = prev_close - rng * 1.1
    
    # Pivot point (close of previous day)
    pivot = prev_close
    
    # Daily trend: EMA50
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: volume > 2.0 * 2-period average (1 day worth at 12h)
    vol_ma_2 = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    volume_spike = volume > 2.0 * vol_ma_2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > R3 + daily uptrend + volume spike
            if close[i] > r3_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < S3 + daily downtrend + volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below pivot (previous day's close)
            if close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above pivot (previous day's close)
            if close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals