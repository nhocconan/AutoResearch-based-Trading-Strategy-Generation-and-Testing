#!/usr/bin/env python3
# 1d_Weekly_Pivot_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Breakout above weekly Camarilla R3 or below S3 in the direction of weekly trend, confirmed by volume spike.
# Weekly pivot levels provide strong support/resistance; breakouts indicate momentum. Weekly trend filter ensures alignment with higher timeframe momentum.
# Volume spike confirms institutional participation. Works in bull (breakouts above R3 in uptrend) and bear (breakdowns below S3 in downtrend).
# Low frequency due to reliance on weekly pivots and strict volume confirmation.

name = "1d_Weekly_Pivot_R3_S3_Breakout_1wTrend_Volume"
timeframe = "1d"
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

    # Get weekly data for Camarilla pivot calculation and trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla levels
    # Using weekly high, low, close
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Pivot point
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    # Camarilla levels
    R3 = weekly_close + (weekly_high - weekly_low) * 1.1 / 2
    S3 = weekly_close - (weekly_high - weekly_low) * 1.1 / 2
    
    # Align weekly levels to daily timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    
    # Weekly trend: EMA50
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike: volume > 2.0 * 20-period average (20 days worth)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or
            np.isnan(pivot_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > weekly R3 + weekly uptrend + volume spike
            if close[i] > R3_aligned[i] and close[i] > ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < weekly S3 + weekly downtrend + volume spike
            elif close[i] < S3_aligned[i] and close[i] < ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below weekly pivot OR trend reversal
            if close[i] < pivot_aligned[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above weekly pivot OR trend reversal
            if close[i] > pivot_aligned[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals