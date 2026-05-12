#!/usr/bin/env python3
# 1d_Camarilla_Pivot_Breakout_WeeklyTrend_Volume
# Hypothesis: Use weekly trend filter with daily Camarilla pivot breakouts.
# Enter long when price breaks above R1 with volume confirmation and weekly uptrend.
# Enter short when price breaks below S1 with volume confirmation and weekly downtrend.
# Exit when price returns to daily pivot or weekly trend reverses.
# Weekly trend filter reduces whipsaw and aligns with major market direction.
# Designed for low trade frequency (target: 10-25 trades/year) to minimize fee drag.
# Works in bull markets (riding breakouts) and bear markets (catching breakdowns with trend filter).

name = "1d_Camarilla_Pivot_Breakout_WeeklyTrend_Volume"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    # Calculate weekly trend: EMA21
    ema_21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)

    # Calculate daily Camarilla levels based on previous day's OHLC
    ph = df_1d['high'].shift(1).values  # Previous day high
    pl = df_1d['low'].shift(1).values   # Previous day low
    pc = df_1d['close'].shift(1).values # Previous day close
    
    pivot = (ph + pl + pc) / 3
    range_val = ph - pl
    
    # Camarilla levels
    r1 = pc + range_val * 1.1 / 12
    r2 = pc + range_val * 1.1 / 6
    r3 = pc + range_val * 1.1 / 4
    r4 = pc + range_val * 1.1 / 2
    s1 = pc - range_val * 1.1 / 12
    s2 = pc - range_val * 1.1 / 6
    s3 = pc - range_val * 1.1 / 4
    s4 = pc - range_val * 1.1 / 2
    
    # Align daily levels to daily timeframe (no shift needed as already daily)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)

    # Volume confirmation: current volume > 1.5x average of last 20 days
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema_21_1w_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Check weekly trend alignment
        price_above_weekly_ema = close[i] > ema_21_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_21_1w_aligned[i]

        if position == 0:
            # LONG: break above R1 with volume and weekly uptrend
            if close[i] > r1_aligned[i] and volume_ok[i] and price_above_weekly_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1 with volume and weekly downtrend
            elif close[i] < s1_aligned[i] and volume_ok[i] and price_below_weekly_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: return to pivot or weekly trend turns down
            if close[i] < pivot_aligned[i] or close[i] < ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: return to pivot or weekly trend turns up
            if close[i] > pivot_aligned[i] or close[i] > ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals