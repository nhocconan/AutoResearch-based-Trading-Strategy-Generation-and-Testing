# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Use Camarilla pivot levels (R1, S1) from the daily chart for breakout entries on 12h. Trade only in the direction of the daily EMA34 trend, with volume confirmation (>1.3x 20-period average volume). Exit when price returns to the daily pivot (PP) or reverses via opposite Camarilla level touch. Designed to work in both bull and bear markets by aligning with the daily trend and requiring volume to avoid false breakouts. Target: 15-30 trades/year.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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

    # Get daily data for Camarilla levels and trend filter (call once before loop)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)

    # Calculate Camarilla levels from previous daily bar
    # Using (H, L, C) of completed daily bar to avoid look-ahead
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values

    # Camarilla levels: based on previous day's range
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    # PP = (H+L+C)/3
    rng = daily_high - daily_low
    camarilla_pp = (daily_high + daily_low + daily_close) / 3.0
    camarilla_r1 = daily_close + rng * 1.1 / 12.0
    camarilla_s1 = daily_close - rng * 1.1 / 12.0

    # Align Camarilla levels to 12h timeframe (available after daily bar close)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_daily, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s1)

    # Daily EMA34 for trend filter
    ema34_daily = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)

    # Volume confirmation: volume > 1.3x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        pp = camarilla_pp_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        ema34_val = ema34_daily_aligned[i]
        vol_avg_val = vol_avg_20[i]

        if np.isnan(pp) or np.isnan(r1) or np.isnan(s1) or np.isnan(ema34_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above R1 + daily uptrend + volume confirmation
            if close[i] > r1 and close[i] > ema34_val and volume[i] > vol_avg_val * 1.3:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1 + daily downtrend + volume confirmation
            elif close[i] < s1 and close[i] < ema34_val and volume[i] > vol_avg_val * 1.3:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to PP or breaks below S1 (reversal)
            if close[i] <= pp or close[i] < s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to PP or breaks above R1 (reversal)
            if close[i] >= pp or close[i] > r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals