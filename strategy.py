#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hTrend_Volume_Filtered
# Hypothesis: Price reacts to Camarilla pivot levels (R1/S1) derived from 12h timeframe.
# Go long when price breaks above R1 with 12h uptrend (EMA50 > EMA200) and volume confirmation.
# Go short when price breaks below S1 with 12h downtrend (EMA50 < EMA200) and volume confirmation.
# Added EMA200 trend filter to reduce false signals and overtrading. Volume spike confirms institutional participation.
# Works in bull markets (breakouts above R1 in uptrend) and bear markets (breakdowns below S1 in downtrend).
# Target: 20-50 trades/year per symbol to minimize fee drift.

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume_Filtered"
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

    # Get 12h data for Camarilla pivot calculation and trend
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla pivot levels (R1, S1) from previous 12h bar
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    camarilla_width = (high_12h - low_12h) * 1.1 / 12
    r1 = close_12h + camarilla_width
    s1 = close_12h - camarilla_width
    
    # 12h trend: EMA50 and EMA200 for stronger filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12h indicators to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    # Volume spike: volume > 2.0 * 3-period average (1.5 days worth at 4h)
    vol_ma_3 = pd.Series(volume).rolling(window=3, min_periods=3).mean().values
    volume_spike = volume > 2.0 * vol_ma_3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(ema200_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > R1 + 12h uptrend (EMA50 > EMA200) + volume spike
            if close[i] > r1_aligned[i] and ema50_12h_aligned[i] > ema200_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < S1 + 12h downtrend (EMA50 < EMA200) + volume spike
            elif close[i] < s1_aligned[i] and ema50_12h_aligned[i] < ema200_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below S1 or trend reversal (EMA50 < EMA200)
            if close[i] < s1_aligned[i] or ema50_12h_aligned[i] < ema200_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above R1 or trend reversal (EMA50 > EMA200)
            if close[i] > r1_aligned[i] or ema50_12h_aligned[i] > ema200_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals