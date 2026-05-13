#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike
# Hypothesis: Camarilla pivot levels (R1/S1) derived from 12h timeframe provide strong intraday support/resistance.
# Go long when price breaks above R1 with 12h trend confirmation (EMA50) and volume spike.
# Go short when price breaks below S1 with 12h trend confirmation and volume spike.
# Exit when price reverts to the pivot point (CP) or trend reverses.
# Uses 12h for pivot calculation and trend filter to avoid lower timeframe noise.
# Volume spike confirms institutional participation. Works in bull/bear by following 12h trend.

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike"
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

    # Get 12h data for Camarilla pivot calculation and trend
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla pivot levels from 12h OHLC
    # CP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    cp_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    r1_12h = df_12h['close'] + (df_12h['high'] - df_12h['low']) * 1.1 / 12
    s1_12h = df_12h['close'] - (df_12h['high'] - df_12h['low']) * 1.1 / 12
    
    # Align pivot levels to 4h timeframe
    cp_12h_aligned = align_htf_to_ltf(prices, df_12h, cp_12h.values)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h.values)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h.values)
    
    # 12h trend: EMA50
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike: volume > 2.0 * 20-period average (~10 days at 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(cp_12h_aligned[i]) or 
            np.isnan(r1_12h_aligned[i]) or 
            np.isnan(s1_12h_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + 12h uptrend + volume spike
            if close[i] > r1_12h_aligned[i] and close[i] > ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + 12h downtrend + volume spike
            elif close[i] < s1_12h_aligned[i] and close[i] < ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to CP or trend turns down
            if close[i] < cp_12h_aligned[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reverts to CP or trend turns up
            if close[i] > cp_12h_aligned[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals