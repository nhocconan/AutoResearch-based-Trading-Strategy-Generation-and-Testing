#!/usr/bin/env python3
# 12h_Weekly_Pivot_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Breakout above weekly Camarilla R3 or below S3 in the direction of daily trend, confirmed by volume spike.
# Weekly pivot levels provide strong support/resistance; breakouts indicate momentum. Daily trend filter ensures alignment with higher timeframe momentum.
# Volume spike confirms institutional participation. Works in bull (breakouts above R3 in uptrend) and bear (breakdowns below S3 in downtrend).
# Low frequency due to reliance on weekly pivots and strict volume confirmation.

name = "12h_Weekly_Pivot_R3_S3_Breakout_1dTrend_Volume"
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

    # Get weekly data for Camarilla pivot calculation
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
    
    # Align weekly levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Daily trend: EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: volume > 2.0 * 24-period average (2 days worth at 12h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * vol_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > weekly R3 + daily uptrend + volume spike
            if close[i] > R3_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < weekly S3 + daily downtrend + volume spike
            elif close[i] < S3_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below weekly pivot OR trend reversal
            pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
            if close[i] < pivot_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above weekly pivot OR trend reversal
            pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
            if close[i] > pivot_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals