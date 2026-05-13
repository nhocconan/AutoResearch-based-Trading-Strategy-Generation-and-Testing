#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: In 12h timeframe, Camarilla pivot levels from 1-day data act as strong support/resistance.
# Go long when price breaks above R1 level with volume spike and 1-day EMA34 uptrend.
# Go short when price breaks below S1 level with volume spike and 1-day EMA34 downtrend.
# Exit when price crosses back through the pivot point or trend changes.
# Uses 1-day trend filter and volume confirmation to reduce false breaks.
# Target: 20-50 trades/year (80-200 total) to avoid fee drag.

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

    # Get 1d data for Camarilla calculation and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each 1d bar
    # R1 = Close + (High - Low) * 1.12
    # S1 = Close - (High - Low) * 1.12
    camarilla_r1 = df_1d['close'] + (df_1d['high'] - df_1d['low']) * 1.12
    camarilla_s1 = df_1d['close'] - (df_1d['high'] - df_1d['low']) * 1.12
    camarilla_pivot = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1.values)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot.values)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: volume > 1.8 * 20-period average (~10 days at 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or 
            np.isnan(ema34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + volume spike + 1d uptrend
            if close[i] > r1_aligned[i] and volume_spike[i] and close[i] > ema34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + volume spike + 1d downtrend
            elif close[i] < s1_aligned[i] and volume_spike[i] and close[i] < ema34_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below pivot or trend turns down
            if close[i] < pivot_aligned[i] or close[i] < ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above pivot or trend turns up
            if close[i] > pivot_aligned[i] or close[i] > ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals