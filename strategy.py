#!/usr/bin/env python3
# 12h_Camarilla_P1_P2_Breakout_1dTrend_Volume
# Hypothesis: Price reacts to Camarilla pivot levels (P1/P2) derived from 1d timeframe.
# Go long when price breaks above P1 with 1d uptrend and volume confirmation.
# Go short when price breaks below P2 with 1d downtrend and volume confirmation.
# P1 = C + (H-L) * 1.1/6, P2 = C - (H-L) * 1.1/6 (wider bands than R1/S1 for fewer, stronger signals).
# Uses 1d trend filter (EMA34) to avoid counter-trend trades. Volume spike confirms institutional participation.
# Designed for low frequency (12-37 trades/year) to minimize fee drag on 12h timeframe.
# Works in bull markets (breakouts above P1 in uptrend) and bear markets (breakdowns below P2 in downtrend).

name = "12h_Camarilla_P1_P2_Breakout_1dTrend_Volume"
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

    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels (P1, P2) from previous 1d bar
    # P1 = C + (H-L) * 1.1/6
    # P2 = C - (H-L) * 1.1/6
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    camarilla_width = (high_1d - low_1d) * 1.1 / 6
    p1 = close_1d + camarilla_width
    p2 = close_1d - camarilla_width
    
    # 1d trend: EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 12h timeframe
    p1_aligned = align_htf_to_ltf(prices, df_1d, p1)
    p2_aligned = align_htf_to_ltf(prices, df_1d, p2)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: volume > 2.0 * 4-period average (2 days worth at 12h)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > 2.0 * vol_ma_4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(p1_aligned[i]) or 
            np.isnan(p2_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > P1 + 1d uptrend + volume spike
            if close[i] > p1_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < P2 + 1d downtrend + volume spike
            elif close[i] < p2_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below P2 or trend reversal
            if close[i] < p2_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above P1 or trend reversal
            if close[i] > p1_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals