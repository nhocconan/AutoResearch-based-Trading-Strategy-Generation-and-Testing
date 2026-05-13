#!/usr/bin/env python3
# 6h_WeeklyPivot_Breakout_1dTrend_Volume
# Hypothesis: Price reacts to weekly pivot levels (R1/S1) derived from weekly timeframe.
# Go long when price breaks above weekly R1 with daily uptrend and volume confirmation.
# Go short when price breaks below weekly S1 with daily downtrend and volume confirmation.
# Weekly pivots provide strong support/resistance in ranging markets, while daily trend filter ensures alignment with higher timeframe momentum.
# Volume spike confirms institutional participation, reducing false breakouts.
# Works in bull markets (breakouts above R1 in uptrend) and bear markets (breakdowns below S1 in downtrend).
# Target: 12-37 trades/year per symbol to minimize fee drag on 6h timeframe.

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

    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot levels (R1, S1) from previous weekly bar
    # R1 = C + (H-L) * 1.1/12
    # S1 = C - (H-L) * 1.1/12
    close_weekly = df_weekly['close'].values
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    weekly_range = high_weekly - low_weekly
    camarilla_width = weekly_range * 1.1 / 12
    r1 = close_weekly + camarilla_width
    s1 = close_weekly - camarilla_width
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    
    # Daily trend: EMA34
    ema34_daily = pd.Series(close_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly and daily indicators to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Volume spike: volume > 2.0 * 3-period average (3 periods at 6h = 18h)
    vol_ma_3 = pd.Series(volume).rolling(window=3, min_periods=3).mean().values
    volume_spike = volume > 2.0 * vol_ma_3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > weekly R1 + daily uptrend + volume spike
            if close[i] > r1_aligned[i] and close[i] > ema34_daily_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < weekly S1 + daily downtrend + volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema34_daily_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below weekly S1 or daily trend reversal
            if close[i] < s1_aligned[i] or close[i] < ema34_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above weekly R1 or daily trend reversal
            if close[i] > r1_aligned[i] or close[i] > ema34_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals