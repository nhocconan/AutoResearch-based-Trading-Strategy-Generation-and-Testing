#!/usr/bin/env python3
# 1d_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Price reacts to Camarilla pivot levels (R1/S1) derived from 1d timeframe. 
# Go long when price breaks above R1 with 1w uptrend and volume confirmation.
# Go short when price breaks below S1 with 1w downtrend and volume confirmation.
# Uses 1d primary timeframe with 1w trend filter to reduce whipsaw and capture major trends.
# Volume spike confirms institutional participation, reducing false breakouts.
# Target: 15-30 trades/year per symbol to minimize fee drift and improve generalization.

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "1d"
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

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d-based Camarilla pivot levels (R1, S1) from previous 1d bar
    # R1 = C + (H-L) * 1.1/12
    # S1 = C - (H-L) * 1.1/12
    close_1d = close  # Using current 1d close for pivot calc (will be lagged via alignment)
    high_1d = high
    low_1d = low
    
    camarilla_width = (high_1d - low_1d) * 1.1 / 12
    r1 = close_1d + camarilla_width
    s1 = close_1d - camarilla_width
    
    # 1w trend: EMA50
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w indicators to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike: volume > 2.0 * 20-day average (to avoid noise)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(r1[i]) or 
            np.isnan(s1[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > R1 + 1w uptrend + volume spike
            if close[i] > r1[i] and close[i] > ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < S1 + 1w downtrend + volume spike
            elif close[i] < s1[i] and close[i] < ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below S1 or trend reversal
            if close[i] < s1[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above R1 or trend reversal
            if close[i] > r1[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals