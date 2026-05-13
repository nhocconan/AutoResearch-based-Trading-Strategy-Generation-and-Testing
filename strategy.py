#!/usr/bin/env python3
# 1d_Pivot_R1_S1_Breakout_WeeklyTrend_Volume
# Hypothesis: Enter long when price breaks above weekly Camarilla R1 level in the direction of weekly EMA20 trend, confirmed by volume spike. Enter short when price breaks below S1 level in downtrend with volume confirmation. Weekly pivot levels provide institutional support/resistance. Trend filter aligns with higher timeframe momentum. Volume spike confirms breakout strength. Works in bull (breakouts above R1 in uptrend) and bear (breakdowns below S1 in downtrend). Low frequency due to pivot level requirement and volume confirmation.

name = "1d_Pivot_R1_S1_Breakout_WeeklyTrend_Volume"
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

    # Get weekly data for pivot levels and trend
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point calculation
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Camarilla levels
    r1 = close_1w + (range_1w * 1.1 / 12)
    s1 = close_1w - (range_1w * 1.1 / 12)
    
    # Weekly trend: EMA20
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly indicators to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume spike: volume > 2.0 * 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > R1 + weekly uptrend + volume spike
            if close[i] > r1_aligned[i] and close[i] > ema20_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < S1 + weekly downtrend + volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema20_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below S1 or trend reversal
            if close[i] < s1_aligned[i] or close[i] < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above R1 or trend reversal
            if close[i] > r1_aligned[i] or close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals