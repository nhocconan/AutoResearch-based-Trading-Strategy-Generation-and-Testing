#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Camarilla pivot levels from daily data act as strong support/resistance. Breakouts above R1 or below S1 with weekly trend confirmation and volume spikes capture momentum moves. Works in bull markets via breakouts and in bear via mean reversion touches of the pivot point (close). Weekly trend filter avoids counter-trend trades. Target: 15-30 trades/year per symbol.

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "12h"
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

    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels (based on previous day)
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    # Using previous day's values to avoid look-ahead
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first period
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]

    range_1d = prev_high - prev_low
    camarilla_r1 = prev_close + range_1d * 1.1 / 12
    camarilla_s1 = prev_close - range_1d * 1.1 / 12

    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Volume confirmation: volume > 1.5x 50-period average
    vol_avg_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        if np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_avg_50[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above Camarilla R1 + weekly uptrend + volume spike
            if close[i] > camarilla_r1_aligned[i] and close[i] > ema50_1w_aligned[i] and volume[i] > vol_avg_50[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below Camarilla S1 + weekly downtrend + volume spike
            elif close[i] < camarilla_s1_aligned[i] and close[i] < ema50_1w_aligned[i] and volume[i] > vol_avg_50[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses below Camarilla pivot point (close) or weekly trend turns down
            pivot_point = (prev_close[i] + prev_high[i] + prev_low[i]) / 3 if i > 0 else (close_1d[0] + high_1d[0] + low_1d[0]) / 3
            # Align pivot point to current index
            if i < len(prev_close):
                pivot_aligned = (prev_close[i] + prev_high[i] + prev_low[i]) / 3
            else:
                pivot_aligned = camarilla_r1_aligned[i]  # fallback
            
            if close[i] < pivot_aligned or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses above Camarilla pivot point or weekly trend turns up
            if i < len(prev_close):
                pivot_aligned = (prev_close[i] + prev_high[i] + prev_low[i]) / 3
            else:
                pivot_aligned = camarilla_s1_aligned[i]  # fallback
                
            if close[i] > pivot_aligned or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals