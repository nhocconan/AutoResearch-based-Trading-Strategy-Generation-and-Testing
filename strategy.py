#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend
# Hypothesis: Enter long when price breaks above Camarilla R3 level on 12h timeframe, with 1d EMA50 trend confirmation and volume spike. Enter short when price breaks below S3 level with 1d EMA50 downtrend and volume spike. Camarilla levels provide institutional support/resistance, trend filter ensures directional bias, volume confirms breakout strength. Works in bull (breakouts above R3 in uptrend) and bear (breakdowns below S3 in downtrend). Low frequency due to specific level breaks and volume confirmation.

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

    # Get daily data for Camarilla calculation and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Camarilla levels from previous day (use previous day's close to avoid look-ahead)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels for TODAY based on YESTERDAY's OHLC
    # Shift by 1 to use previous day's data
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]  # first day fallback
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Camarilla calculations
    range_ = prev_high - prev_low
    camarilla_base = prev_close
    camarilla_r3 = camarilla_base + 1.1 * range_ / 2
    camarilla_s3 = camarilla_base - 1.1 * range_ / 2
    
    # Daily trend: EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
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
            # EXIT LONG: Close below S3 or trend reversal
            if close[i] < s3_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above R3 or trend reversal
            if close[i] > r3_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals