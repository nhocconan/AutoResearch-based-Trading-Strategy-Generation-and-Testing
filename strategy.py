#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Camarilla pivot breakout on 12h, filtered by 1w trend and volume spikes.
# Uses daily Camarilla levels (R1, S1) from prior day as support/resistance.
# Trend filter: 1w EMA50 (only trade in direction of higher timeframe trend).
# Volume confirmation: current volume > 2.0 x 20-period average.
# Designed to work in both bull and bear markets by following 1w trend direction.
# Target: 12-37 trades/year per symbol to minimize fee drag.

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

    # Get daily data for Camarilla levels (prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')

    # Calculate Camarilla levels from prior day's OHLC
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = df_1d['close'] + (df_1d['high'] - df_1d['low']) * 1.1 / 12
    camarilla_s1 = df_1d['close'] - (df_1d['high'] - df_1d['low']) * 1.1 / 12
    camarilla_r1_vals = camarilla_r1.values
    camarilla_s1_vals = camarilla_s1.values

    # Align Camarilla levels to 12h timeframe (use prior day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_vals)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_vals)

    # Trend filter: 1w EMA50
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above Camarilla R1 in uptrend with volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Camarilla S1 in downtrend with volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S1 or trend turns down
            if close[i] < camarilla_s1_aligned[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R1 or trend turns up
            if close[i] > camarilla_r1_aligned[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals