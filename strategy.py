#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_R4_S4_Breakout_WeeklyTrend_VolumeSpike
Hypothesis: Price breaking above/below Camarilla R4/S4 levels (derived from weekly high-low-close) with weekly EMA trend filter and volume confirmation (2x average) captures strong trending moves while avoiding false breakouts. R4/S4 levels represent stronger support/resistance, reducing false signals. Works in bull/bear by following weekly trend direction. Designed for low trade frequency (12-37/year) to minimize fee drag.
"""

name = "12h_Camarilla_Pivot_R4_S4_Breakout_WeeklyTrend_VolumeSpike"
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

    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')

    # Calculate Camarilla levels from weekly data
    # Camarilla: R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    # where C = close, H = high, L = low of previous week
    close_weekly = df_weekly['close'].values
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values

    # Shift by 1 to use previous week's data
    prev_close_weekly = np.roll(close_weekly, 1)
    prev_high_weekly = np.roll(high_weekly, 1)
    prev_low_weekly = np.roll(low_weekly, 1)
    prev_close_weekly[0] = np.nan
    prev_high_weekly[0] = np.nan
    prev_low_weekly[0] = np.nan

    camarilla_upper = prev_close_weekly + (prev_high_weekly - prev_low_weekly) * 1.1 / 2
    camarilla_lower = prev_close_weekly - (prev_high_weekly - prev_low_weekly) * 1.1 / 2

    # Align Camarilla levels to 12h timeframe
    camarilla_upper_aligned = align_htf_to_ltf(prices, df_weekly, camarilla_upper)
    camarilla_lower_aligned = align_htf_to_ltf(prices, df_weekly, camarilla_lower)

    # Weekly EMA34 trend filter
    ema_34_weekly = pd.Series(close_weekly).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_34_weekly)

    # Volume spike: >2x 24-period average (12h, ~2 weeks)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):  # Start after EMA34 warmup
        if (np.isnan(camarilla_upper_aligned[i]) or np.isnan(camarilla_lower_aligned[i]) or 
            np.isnan(ema_34_weekly_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Camarilla R4 + weekly EMA34 uptrend + volume spike
            if (close[i] > camarilla_upper_aligned[i] and 
                close[i] > ema_34_weekly_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S4 + weekly EMA34 downtrend + volume spike
            elif (close[i] < camarilla_lower_aligned[i] and 
                  close[i] < ema_34_weekly_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Camarilla S4 (reversal level)
            if close[i] < camarilla_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Camarilla R4 (reversal level)
            if close[i] > camarilla_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals