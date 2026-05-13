#!/usr/bin/env python3
# 4H_Camarilla_R1_S1_Breakout_1DTrend_VolumeSpike
# Hypothesis: Price breaks above/below Camarilla R1/S1 levels from the previous day, confirmed by 1d EMA50 trend and volume spike. Works in bull (breakouts above R1 in uptrend) and bear (breakdowns below S1 in downtrend). Uses 4h timeframe for low trade frequency and high edge.

name = "4H_Camarilla_R1_S1_Breakout_1DTrend_VolumeSpike"
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

    # Get daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Previous day's high, low, close for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Daily trend: EMA50
    ema50_1d = pd.Series(prev_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: volume > 2.0 * 6-period average (1.5 days worth at 4h)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_spike = volume > 2.0 * vol_ma_6
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > R1 + daily uptrend + volume spike
            if close[i] > R1_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < S1 + daily downtrend + volume spike
            elif close[i] < S1_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below EMA50
            if close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above EMA50
            if close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals