#!/usr/bin/env python3
# 1d_WeeklyPivot_Breakout_Trend_Volume
# Hypothesis: Price reacts to weekly pivot levels (R1/S1) derived from 1w timeframe.
# Go long when price breaks above R1 with 1w uptrend and volume confirmation.
# Go short when price breaks below S1 with 1w downtrend and volume confirmation.
# Weekly pivots provide robust support/resistance; trend filter ensures alignment with higher timeframe momentum.
# Volume spike confirms institutional participation, reducing false breakouts.
# Designed for low trade frequency (<25/year) to minimize fee drag in ranging/bear markets.

name = "1d_WeeklyPivot_Breakout_Trend_Volume"
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

    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot levels (R1, S1) from previous weekly bar
    # R1 = C + (H-L) * 1.1/12
    # S1 = C - (H-L) * 1.1/12
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    pivot_width = (high_1w - low_1w) * 1.1 / 12
    r1 = close_1w + pivot_width
    s1 = close_1w - pivot_width
    
    # Weekly trend: EMA50
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike: volume > 2.0 * 10-period average (approx 2 weeks)
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_spike = volume > 2.0 * vol_ma_10
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > R1 + weekly uptrend + volume spike
            if close[i] > r1_aligned[i] and close[i] > ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < S1 + weekly downtrend + volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below S1 or trend reversal
            if close[i] < s1_aligned[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above R1 or trend reversal
            if close[i] > r1_aligned[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals