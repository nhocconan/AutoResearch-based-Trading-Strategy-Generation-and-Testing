#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike
# Hypothesis: Breakout above weekly Camarilla R1 or below S1 in the direction of weekly trend, confirmed by volume spike.
# Weekly trend filter ensures alignment with higher timeframe momentum. Volume spike confirms institutional participation.
# Works in bull (breakouts above R1 in uptrend) and bear (breakdowns below S1 in downtrend).
# Low frequency due to reliance on weekly pivots and strict volume confirmation.

name = "4h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike"
timeframe = "4h"
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
    # Camarilla levels R1 and S1
    R1 = weekly_close + (weekly_high - weekly_low) * 1.1 / 12
    S1 = weekly_close - (weekly_high - weekly_low) * 1.1 / 12
    
    # Align weekly levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    
    # Get weekly trend: EMA50
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike: volume > 2.0 * 6-period average (1 day worth at 4h)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_spike = volume > 2.0 * vol_ma_6
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or
            np.isnan(pivot_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > weekly R1 + weekly uptrend + volume spike
            if close[i] > R1_aligned[i] and close[i] > ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < weekly S1 + weekly downtrend + volume spike
            elif close[i] < S1_aligned[i] and close[i] < ema50_1w_aligned[i] and volume_spike[i]:
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