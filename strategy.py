#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_Filtered
Hypothesis: On 1h timeframe, buy when price breaks above Camarilla R1 from previous 4h with volume >1.5x average and 4h EMA34 trending up; sell when price breaks below Camarilla S1 with volume >1.5x average and 4h EMA34 trending down. Session filter (08-20 UTC) reduces noise. Targets 15-35 trades per year to minimize fee drag and improve generalization in bull/bear markets.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Filtered"
timeframe = "1h"
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

    # Get 4h data for Camarilla levels and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values

    # Calculate Camarilla levels from previous 4h bar
    # Camarilla: R1 = close + (high - low) * 1.12/12, S1 = close - (high - low) * 1.12/12
    range_4h = high_4h - low_4h
    camarilla_r1 = close_4h + range_4h * 1.12 / 12
    camarilla_s1 = close_4h - range_4h * 1.12 / 12

    # Use previous 4h bar's levels (shift by 1)
    camarilla_r1_prev = np.roll(camarilla_r1, 1)
    camarilla_s1_prev = np.roll(camarilla_s1, 1)
    camarilla_r1_prev[0] = np.nan
    camarilla_s1_prev[0] = np.nan

    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1_prev)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1_prev)

    # 4h EMA34 for trend filter
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)

    # Volume confirmation: volume > 1.5x 24-period average (approx 1 day)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN or outside session
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema34_4h_aligned[i]) or np.isnan(vol_avg_24[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Camarilla R1 + 4h uptrend + volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema34_4h_aligned[i] and 
                volume[i] > vol_avg_24[i] * 1.5):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below Camarilla S1 + 4h downtrend + volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema34_4h_aligned[i] and 
                  volume[i] > vol_avg_24[i] * 1.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S1 OR trend turns down
            if close[i] < camarilla_s1_aligned[i] or close[i] < ema34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R1 OR trend turns up
            if close[i] > camarilla_r1_aligned[i] or close[i] > ema34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals