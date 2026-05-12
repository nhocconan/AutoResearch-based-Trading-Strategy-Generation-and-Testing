#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_Volume
Hypothesis: Combine daily Camarilla pivot levels (R1/S1) with weekly trend filter (EMA34) and volume confirmation to capture breakouts with institutional levels. Weekly trend filter reduces false signals in chop, suitable for both bull and bear markets. Target: 15-30 trades/year.
"""

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
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

    # Get daily data for Camarilla pivot levels (R1, S1) - calculate once per day
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each daily bar
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Using previous day's OHLC to avoid look-ahead
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    camarilla_r1 = daily_close + (daily_high - daily_low) * 1.1 / 12
    camarilla_s1 = daily_close - (daily_high - daily_low) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (shifted by 1 day to avoid look-ahead)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r1, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s1, additional_delay_bars=1)

    # Get weekly data for trend filter (EMA34) - use weekly close
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 34:
        return np.zeros(n)
    weekly_close = df_weekly['close'].values
    ema34_weekly = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly, additional_delay_bars=1)

    # Volume confirmation: volume > 1.5x 20-period average on 12h chart
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        ema34_val = ema34_weekly_aligned[i]
        vol_avg_val = vol_avg_20[i]

        if np.isnan(r1) or np.isnan(s1) or np.isnan(ema34_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above R1 + weekly uptrend + volume confirmation
            if close[i] > r1 and close[i-1] <= r1 and close[i] > ema34_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S1 + weekly downtrend + volume confirmation
            elif close[i] < s1 and close[i-1] >= s1 and close[i] < ema34_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close breaks below S1 or weekly trend turns down
            if close[i] < s1 or close[i] < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close breaks above R1 or weekly trend turns up
            if close[i] > r1 or close[i] > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals