#!/usr/bin/env python3
# 6h_WeeklyPivot_Breakout_1dTrend_Volume
# Hypothesis: Price reacts to weekly pivot levels (R4/S4) derived from weekly timeframe.
# Go long when price breaks above R4 with 1-day uptrend and volume confirmation.
# Go short when price breaks below S4 with 1-day downtrend and volume confirmation.
# Weekly pivots capture key institutional levels; daily trend ensures alignment with intermediate momentum.
# Volume spike confirms institutional participation, reducing false breakouts.
# Designed for 6h timeframe to balance trade frequency (~15-35/year) and capture meaningful moves.
# Works in bull markets (breakouts above R4 in uptrend) and bear markets (breakdowns below S4 in downtrend).

name = "6h_WeeklyPivot_Breakout_1dTrend_Volume"
timeframe = "6h"
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

    # Get weekly data for pivot calculation (previous week's close)
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot levels (using prior week's OHLC)
    # Pivot Point (PP) = (High + Low + Close) / 3
    # R4 = Close + 3*(High - Low)  (aggressive breakout level)
    # S4 = Close - 3*(High - Low)  (aggressive breakdown level)
    weekly_close = df_weekly['close'].values
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    weekly_range = weekly_high - weekly_low
    r4 = weekly_close + 3 * weekly_range
    s4 = weekly_close - 3 * weekly_range
    
    # 1-day trend: EMA34 (standard for trend identification)
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly and daily indicators to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_weekly, r4)
    s4_aligned = align_htf_to_ltf(prices, df_weekly, s4)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: volume > 2.5 * 20-period average (approx 5 days at 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > R4 + 1-day uptrend + volume spike
            if close[i] > r4_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < S4 + 1-day downtrend + volume spike
            elif close[i] < s4_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below S4 or trend reversal
            if close[i] < s4_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above R4 or trend reversal
            if close[i] > r4_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals