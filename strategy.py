#!/usr/bin/env python3
# 1d_Camarilla_R1_S1_Breakout_WeeklyTrend_Volume
# Hypothesis: Daily Camarilla R1/S1 breakout with weekly trend filter (price above/below weekly EMA34) and volume confirmation.
# Weekly trend filter reduces false breakouts in sideways markets by ensuring alignment with higher timeframe momentum.
# Works in bull (follows breakouts with bullish weekly trend) and bear (avoids bullish breakouts in bearish weekly trend).
# Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_Camarilla_R1_S1_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
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

    # Get daily data for Camarilla calculation (using same timeframe)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for R1 and S1 using previous day's OHLC
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_close = df_1d['close'].shift(1)
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12.0
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12.0
    
    # Get weekly data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    weekly_ema34 = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_ema34_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema34)

    # Volume filter: >1.5x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(weekly_ema34_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above Camarilla R1 + price above weekly EMA34 (bullish trend) + volume spike
            if (close[i] > camarilla_r1[i] and 
                close[i] > weekly_ema34_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Camarilla S1 + price below weekly EMA34 (bearish trend) + volume spike
            elif (close[i] < camarilla_s1[i] and 
                  close[i] < weekly_ema34_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Camarilla S1 or weekly trend turns bearish
            if (close[i] < camarilla_s1[i] or close[i] < weekly_ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above Camarilla R1 or weekly trend turns bullish
            if (close[i] > camarilla_r1[i] or close[i] > weekly_ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals