#!/usr/bin/env python3
"""
6h_Weekly_Camarilla_R1S1_Breakout_1dTrend_Volume
Hypothesis: Breakout above weekly R1 or below weekly S1 with daily EMA34 trend filter and volume confirmation. Designed for 10-30 trades/year on 6h timeframe to work in both bull and bear markets by using strong weekly institutional levels and filtering with daily trend and volume. Weekly levels provide more robust support/resistance than daily, reducing false breakouts while maintaining trend alignment.
"""

name = "6h_Weekly_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "6h"
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
    if len(df_weekly) < 2:
        return np.zeros(n)

    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values

    # Calculate Weekly Camarilla pivot levels for previous week
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    hl_range_weekly = high_weekly - low_weekly
    r1_weekly = close_weekly + hl_range_weekly * 1.1 / 12
    s1_weekly = close_weekly - hl_range_weekly * 1.1 / 12

    # Align to 6h timeframe (values from previous week's close)
    r1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r1_weekly)
    s1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s1_weekly)

    # Get daily EMA34 for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    close_daily = df_daily['close'].values
    ema34_daily = pd.Series(close_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)

    # Volume confirmation: volume > 1.5x 20-period average (on 6h timeframe)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        r1_val = r1_weekly_aligned[i]
        s1_val = s1_weekly_aligned[i]
        ema34_val = ema34_daily_aligned[i]
        vol_avg_val = vol_avg_20[i]

        if np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(ema34_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above weekly R1 + daily uptrend + volume confirmation
            if close[i] > r1_val and close[i] > ema34_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Close below weekly S1 + daily downtrend + volume confirmation
            elif close[i] < s1_val and close[i] < ema34_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below daily EMA34 or weekly S3 (strong reversal)
            camarilla_s3_weekly = close_weekly - hl_range_weekly * 1.1 / 4
            s3_weekly_aligned = align_htf_to_ltf(prices, df_weekly, 
                                np.full_like(close_weekly, camarilla_s3_weekly))
            if close[i] < ema34_val or close[i] < s3_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above daily EMA34 or weekly R3 (strong reversal)
            camarilla_r3_weekly = close_weekly + hl_range_weekly * 1.1 / 4
            r3_weekly_aligned = align_htf_to_ltf(prices, df_weekly, 
                                np.full_like(close_weekly, camarilla_r3_weekly))
            if close[i] > ema34_val or close[i] > r3_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals