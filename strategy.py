#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirmation
# Hypothesis: Camarilla pivot levels (R1/S1) on 4h combined with daily trend and volume
# confirmation capture high-probability breakouts with minimal false signals. Uses
# precise pivot levels for entry and exit, reducing whipsaw. Designed to work in
# both bull and bear markets by filtering with daily trend.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate daily EMA20 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Calculate 4h Camarilla levels (based on previous day's range)
    # For 4h chart, we need daily high/low/close from previous day
    # Since we're using 1d data, we shift it by 1 to avoid look-ahead
    prev_day_high = np.roll(close_1d, 1)  # Simplified: using close as proxy for range
    prev_day_low = np.roll(close_1d, 1)
    prev_day_close = np.roll(close_1d, 1)
    
    # Calculate Camarilla levels for each 4h bar based on previous day
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    # We'll approximate using daily range
    daily_range = np.maximum(prev_day_high, prev_day_close) - np.minimum(prev_day_low, prev_day_close)
    r1_level = prev_day_close + 1.1 * daily_range / 12
    s1_level = prev_day_close - 1.1 * daily_range / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_level)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_level)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 with volume spike and daily uptrend
            if close[i] > r1_aligned[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume spike and daily downtrend
            elif close[i] < s1_aligned[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R1 or daily trend turns down
            if close[i] < r1_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S1 or daily trend turns up
            if close[i] > s1_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals