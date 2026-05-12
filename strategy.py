#!/usr/bin/env python3
# 6h_1W_1D_Camarilla_R4S4_Breakout_Trend_Volume
# Hypothesis: Breakouts at weekly/bi-weekly Camarilla R4/S4 levels with 1w trend filter and volume confirmation.
# Targets major trend continuation moves in both bull and bear markets. R4/S4 represent strong breakout levels.
# Uses weekly trend to avoid counter-trend trades, volume to confirm validity. Designed for 6h to limit trades.

name = "6h_1W_1D_Camarilla_R4S4_Breakout_Trend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter and Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Weekly EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # Get daily data for Camarilla levels (using weekly OHLC for R4/S4)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate weekly Camarilla R4 and S4 levels from previous weekly OHLC
    # Using previous week's data to avoid look-ahead
    prev_weekly_close = df_1w['close'].shift(1).values
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values

    # Camarilla R4 and S4 levels (stronger breakout levels)
    # R4 = C + (H-L) * 1.1/2, S4 = C - (H-L) * 1.1/2
    camarilla_r4 = prev_weekly_close + (prev_weekly_high - prev_weekly_low) * 1.1 / 2
    camarilla_s4 = prev_weekly_close - (prev_weekly_high - prev_weekly_low) * 1.1 / 2

    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)

    # Volume confirmation: current volume > 2.0x average of last 28 periods (approx 1 week)
    vol_ma = pd.Series(volume).rolling(window=28, min_periods=28).mean().values
    volume_ok = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from weekly EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]

        if position == 0:
            # LONG: Break above Camarilla R4 in uptrend with volume confirmation
            if (close[i] > camarilla_r4_aligned[i] and uptrend and volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Camarilla S4 in downtrend with volume confirmation
            elif (close[i] < camarilla_s4_aligned[i] and downtrend and volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R4 or trend reversal
            if close[i] < camarilla_r4_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S4 or trend reversal
            if close[i] > camarilla_s4_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals