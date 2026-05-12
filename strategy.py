#!/usr/bin/env python3
"""
1d_Weekly_Camarilla_R1_S1_Breakout_WeeklyTrend_Volume
Hypothesis: Breakout above weekly R1 or below weekly S1 with weekly EMA20 trend filter and daily volume confirmation. Designed for 7-25 trades/year on 1d timeframe to work in both bull and bear markets by using strong weekly institutional levels and filtering with weekly trend and volume.
"""

name = "1d_Weekly_Camarilla_R1_S1_Breakout_WeeklyTrend_Volume"
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

    # Get weekly data (call once before loop)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)

    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values

    # Calculate Weekly Camarilla pivot levels for previous week
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    hl_range_weekly = high_weekly - low_weekly
    r1_weekly = close_weekly + hl_range_weekly * 1.1 / 12
    s1_weekly = close_weekly - hl_range_weekly * 1.1 / 12

    # Align to daily timeframe (values from previous week's close)
    r1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r1_weekly)
    s1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s1_weekly)

    # Get weekly EMA20 for trend filter
    ema20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)

    # Daily volume confirmation: volume > 1.5x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        r1_val = r1_weekly_aligned[i]
        s1_val = s1_weekly_aligned[i]
        ema20_val = ema20_weekly_aligned[i]
        vol_avg_val = vol_avg_20[i]

        if np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(ema20_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above weekly R1 + weekly uptrend + volume confirmation
            if close[i] > r1_val and close[i] > ema20_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Close below weekly S1 + weekly downtrend + volume confirmation
            elif close[i] < s1_val and close[i] < ema20_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below weekly EMA20 or weekly S3 (strong reversal)
            camarilla_s3_weekly = close_weekly - (high_weekly - low_weekly) * 1.1 / 4
            s3_weekly_aligned = align_htf_to_ltf(prices, df_weekly, 
                                np.full_like(close_weekly, camarilla_s3_weekly))
            if close[i] < ema20_val or close[i] < s3_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above weekly EMA20 or weekly R3 (strong reversal)
            camarilla_r3_weekly = close_weekly + (high_weekly - low_weekly) * 1.1 / 4
            r3_weekly_aligned = align_htf_to_ltf(prices, df_weekly, 
                                np.full_like(close_weekly, camarilla_r3_weekly))
            if close[i] > ema20_val or close[i] > r3_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals