#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_Volume
Hypothesis: Breakout above daily R1 or below daily S1 with 12h KAMA trend filter and volume confirmation. Designed for 12-37 trades/year on 12h timeframe to work in both bull and bear markets by using strong daily institutional levels and filtering with 12h trend and volume. Daily levels provide more robust support/resistance than weekly, reducing false breakouts while maintaining trend alignment.
"""

name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
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

    # Get daily data (call once before loop)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)

    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values

    # Calculate Daily Camarilla pivot levels for previous day
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    hl_range_daily = high_daily - low_daily
    r1_daily = close_daily + hl_range_daily * 1.1 / 12
    s1_daily = close_daily - hl_range_daily * 1.1 / 12

    # Align to 12h timeframe (values from previous day's close)
    r1_daily_aligned = align_htf_to_ltf(prices, df_daily, r1_daily)
    s1_daily_aligned = align_htf_to_ltf(prices, df_daily, s1_daily)

    # Get 12h KAMA for trend filter
    if len(close) < 30:
        return np.zeros(n)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), 
                       mode='same')  # simplified ER approximation
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # kama smoothing constant
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama = kama  # already aligned to 12h

    # Volume confirmation: volume > 1.5x 20-period average (on 12h timeframe)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        r1_val = r1_daily_aligned[i]
        s1_val = s1_daily_aligned[i]
        kama_val = kama[i]
        vol_avg_val = vol_avg_20[i]

        if np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(kama_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above daily R1 + 12h uptrend + volume confirmation
            if close[i] > r1_val and close[i] > kama_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Close below daily S1 + 12h downtrend + volume confirmation
            elif close[i] < s1_val and close[i] < kama_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below 12h KAMA or daily S3 (strong reversal)
            camarilla_s3_daily = close_daily - hl_range_daily * 1.1 / 4
            s3_daily_aligned = align_htf_to_ltf(prices, df_daily, 
                                np.full_like(close_daily, camarilla_s3_daily))
            if close[i] < kama_val or close[i] < s3_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above 12h KAMA or daily R3 (strong reversal)
            camarilla_r3_daily = close_daily + hl_range_daily * 1.1 / 4
            r3_daily_aligned = align_htf_to_ltf(prices, df_daily, 
                                np.full_like(close_daily, camarilla_r3_daily))
            if close[i] > kama_val or close[i] > r3_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals