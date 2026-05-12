#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS
Hypothesis: Price breaking above/below Camarilla R1/S1 levels (derived from 12h high-low-close) with 12h EMA trend filter and volume confirmation (1.5x average) captures strong trending moves while avoiding false breakouts. R1/S1 levels provide more frequent but still significant support/resistance. Works in bull/bear by following 12h trend direction.
"""

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS"
timeframe = "4h"
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

    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')

    # Calculate Camarilla levels from 12h data
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values

    # Shift by 1 to use previous 12h bar's data
    prev_close_12h = np.roll(close_12h, 1)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h[0] = np.nan
    prev_high_12h[0] = np.nan
    prev_low_12h[0] = np.nan

    camarilla_upper = prev_close_12h + (prev_high_12h - prev_low_12h) * 1.1 / 4
    camarilla_lower = prev_close_12h - (prev_high_12h - prev_low_12h) * 1.1 / 4

    # Align Camarilla levels to 4h timeframe
    camarilla_upper_aligned = align_htf_to_ltf(prices, df_12h, camarilla_upper)
    camarilla_lower_aligned = align_htf_to_ltf(prices, df_12h, camarilla_lower)

    # 12h EMA34 trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)

    # Volume spike: >1.5x 20-period average (4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):  # Start after EMA34 warmup
        if (np.isnan(camarilla_upper_aligned[i]) or np.isnan(camarilla_lower_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Camarilla R1 + 12h EMA34 uptrend + volume spike
            if (close[i] > camarilla_upper_aligned[i] and 
                close[i] > ema_34_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S1 + 12h EMA34 downtrend + volume spike
            elif (close[i] < camarilla_lower_aligned[i] and 
                  close[i] < ema_34_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Camarilla S1 (reversal level)
            if close[i] < camarilla_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Camarilla R1 (reversal level)
            if close[i] > camarilla_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals