#!/usr/bin/env python3
# 12h_Camarilla_Pivot_Breakout_1wTrend_Volume
# Hypothesis: Price reacts to Camarilla pivot levels (R3/S3) from the previous 1-week candle.
# Go long when price breaks above R3 with 1-week uptrend and volume confirmation.
# Go short when price breaks below S3 with 1-week downtrend and volume confirmation.
# Weekly trend filter ensures alignment with higher timeframe momentum, reducing false signals.
# Volume spike confirms institutional participation.
# Designed for 12h timeframe to target 12-37 trades/year (50-150 total over 4 years).
# Works in bull markets (breakouts above R3 in uptrend) and bear markets (breakdowns below S3 in downtrend).

name = "12h_Camarilla_Pivot_Breakout_1wTrend_Volume"
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

    # Get 1w data for Camarilla pivot calculation and trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Camarilla pivot levels (R3, S3) from previous 1w bar
    # R3 = C + (H-L) * 1.1/4
    # S3 = C - (H-L) * 1.1/4
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    camarilla_width = (high_1w - low_1w) * 1.1 / 4
    r3 = close_1w + camarilla_width
    s3 = close_1w - camarilla_width
    
    # 1w trend: EMA34
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w indicators to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike: volume > 2.0 * 24-period average (12 days worth at 12h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * vol_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > R3 + 1w uptrend + volume spike
            if close[i] > r3_aligned[i] and close[i] > ema34_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < S3 + 1w downtrend + volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema34_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below S3 or trend reversal
            if close[i] < s3_aligned[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above R3 or trend reversal
            if close[i] > r3_aligned[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals