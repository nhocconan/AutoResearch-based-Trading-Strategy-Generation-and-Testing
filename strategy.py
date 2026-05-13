#!/usr/bin/env python3
# 1d_WeeklyPivot_Camarilla_Breakout_TrendFilter_Volume
# Hypothesis: Combining weekly Pivot/Camarilla levels with daily trend and volume confirmation reduces false breakouts and captures strong directional moves. The weekly timeframe provides robust structure while daily trend filters ensure alignment with higher timeframe momentum. Volume confirmation ensures breakouts have institutional participation. Designed to work in both bull and bear markets by using symmetric long/short logic with proper risk controls. Target: 15-25 trades per year to minimize fee drag and improve generalization.

name = "1d_WeeklyPivot_Camarilla_Breakout_TrendFilter_Volume"
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

    # ATR for volatility context
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values

    # Get weekly data for Pivot and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Pivot and Camarilla levels from previous week
    # Using typical pivot: (H+L+C)/3
    # R1 = Pivot + (H-L)*1.1/12, S1 = Pivot - (H-L)*1.1/12
    prev_week_close = df_1w['close'].shift(1).values
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    weekly_r1 = weekly_pivot + (prev_week_high - prev_week_low) * 1.1 / 12
    weekly_s1 = weekly_pivot - (prev_week_high - prev_week_low) * 1.1 / 12
    
    # Align to daily timeframe (available after previous week close)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)

    # Daily EMA50 for trend filter
    ema50_d = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values

    # Volume filter: >1.5x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(ema50_d[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above weekly R1 + daily EMA50 uptrend + volume spike
            if (close[i] > weekly_r1_aligned[i] and 
                close[i] > ema50_d[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Close below weekly S1 + daily EMA50 downtrend + volume spike
            elif (close[i] < weekly_s1_aligned[i] and 
                  close[i] < ema50_d[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below weekly S1 or trend reversal
            if close[i] < weekly_s1_aligned[i] or close[i] < ema50_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above weekly R1 or trend reversal
            if close[i] > weekly_r1_aligned[i] or close[i] > ema50_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals