#!/usr/bin/env python3
# 6h_ElderRay_WeeklyTrend_Filter
# Hypothesis: Use Elder Ray (Bull/Bear Power) on 6h with weekly trend filter to capture trend continuations.
# Long when Bull Power > 0 and price above weekly EMA50; Short when Bear Power < 0 and price below weekly EMA50.
# Designed to work in both bull and bear markets by aligning with higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

name = "6h_ElderRay_WeeklyTrend_Filter"
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

    # Get 6h data for Elder Ray calculation
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values

    # Calculate EMA13 for Elder Ray (standard period)
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values

    # Bull Power = High - EMA13
    bull_power = high_6h - ema13_6h
    # Bear Power = Low - EMA13
    bear_power = low_6h - ema13_6h

    # Weekly trend filter: EMA50 on 1w
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Align 6h indicators to lower timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):  # Start after warmup period
        # Skip if any required value is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Elder Ray signals
        bull_power_curr = bull_power_aligned[i]
        bear_power_curr = bear_power_aligned[i]
        
        # Price relative to weekly EMA50
        price_above_weekly_ema = close[i] > ema50_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema50_1w_aligned[i]

        if position == 0:
            # LONG: Bull Power > 0 AND price above weekly EMA50 (uptrend)
            if bull_power_curr > 0 and price_above_weekly_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 AND price below weekly EMA50 (downtrend)
            elif bear_power_curr < 0 and price_below_weekly_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power <= 0 OR price below weekly EMA50
            if bull_power_curr <= 0 or not price_above_weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power >= 0 OR price above weekly EMA50
            if bear_power_curr >= 0 or not price_below_weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals