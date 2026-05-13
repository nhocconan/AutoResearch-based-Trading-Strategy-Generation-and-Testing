#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm
# Hypothesis: Breakouts of Camarilla R1 (long) and S1 (short) levels on 1d, 
# confirmed by weekly trend (EMA50 > EMA200) and volume spike (>2x 4-bar average).
# Uses 12h timeframe to reduce trade frequency and avoid fee drag.
# Works in bull (long on R1 breakout in uptrend) and bear (short on S1 breakdown in downtrend).
# Volume spike ensures institutional participation; weekly trend filter avoids counter-trend trades.

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm"
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

    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    weekly_uptrend = ema50_1w > ema200_1w
    
    # Align indicators to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    
    # Volume spike: volume > 2.0 * 4-period average (2 days worth at 12h)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > 2.0 * vol_ma_4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(weekly_uptrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > R1 + weekly uptrend + volume spike
            if close[i] > r1_aligned[i] and weekly_uptrend_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < S1 + weekly downtrend + volume spike
            elif close[i] < s1_aligned[i] and not weekly_uptrend_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below S1 or weekly trend reversal to down
            if close[i] < s1_aligned[i] or not weekly_uptrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above R1 or weekly trend reversal to up
            if close[i] > r1_aligned[i] or weekly_uptrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals