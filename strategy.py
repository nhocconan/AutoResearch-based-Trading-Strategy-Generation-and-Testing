#!/usr/bin/env python3
# 1d_Weekly_Camarilla_R1S1_Breakout_Trend_Filter
# Hypothesis: Breakouts from weekly Camarilla R1/S1 levels on daily chart with weekly trend filter
# and volume confirmation capture institutional moves while avoiding whipsaw in ranging markets.
# Weekly trend filter reduces false breakouts; volume confirms institutional participation.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years).

name = "1d_Weekly_Camarilla_R1S1_Breakout_Trend_Filter"
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

    # Get weekly data for Camarilla levels and trend filter (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels from previous week
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C = (H+L+CLOSE)/3 of previous week
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Previous week's values for current week's levels
    prev_weekly_high = np.roll(weekly_high, 1)
    prev_weekly_low = np.roll(weekly_low, 1)
    prev_weekly_close = np.roll(weekly_close, 1)
    
    # Pivot point (average of H, L, C from previous week)
    pp = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    # Range
    weekly_range = prev_weekly_high - prev_weekly_low
    
    # Camarilla R1 and S1 levels
    r1 = pp + (weekly_range * 1.1 / 12)
    s1 = pp - (weekly_range * 1.1 / 12)
    
    # Align weekly levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Weekly trend filter: EMA20 on weekly close
    weekly_ema20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema20_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema20)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(weekly_ema20_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above weekly R1 + weekly uptrend + volume spike
            if close[i] > r1_aligned[i] and close[i] > weekly_ema20_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below weekly S1 + weekly downtrend + volume spike
            elif close[i] < s1_aligned[i] and close[i] < weekly_ema20_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses below weekly S1 or weekly trend turns down
            if close[i] < s1_aligned[i] or close[i] < weekly_ema20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses above weekly R1 or weekly trend turns up
            if close[i] > r1_aligned[i] or close[i] > weekly_ema20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals