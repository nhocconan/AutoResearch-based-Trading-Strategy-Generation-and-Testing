#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla R1/S1 levels from daily chart act as key support/resistance.
# Breakout above R1 with daily trend confirmation and volume spike triggers long.
# Breakdown below S1 with daily trend confirmation and volume spike triggers short.
# Daily trend uses EMA34 to avoid whipsaw. Works in bull/bear by following higher timeframe trend.
# Target: 20-50 trades/year on 4h with disciplined entries to avoid fee drag.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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

    # Get 1d data for Camarilla and EMA34
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels for previous day
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_r1 = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low']) / 12
    camarilla_s1 = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low']) / 12
    
    # Align 1d indicators to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1.values)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1.values)
    
    # Volume spike: volume > 2.0 * 20-period average (~5 days at 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + daily uptrend + volume spike
            if close[i] > camarilla_r1_aligned[i] and close[i-1] <= camarilla_r1_aligned[i-1] and ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + daily downtrend + volume spike
            elif close[i] < camarilla_s1_aligned[i] and close[i-1] >= camarilla_s1_aligned[i-1] and ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or daily trend turns down
            if close[i] < camarilla_s1_aligned[i] or ema34_1d_aligned[i] < ema34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or daily trend turns up
            if close[i] > camarilla_r1_aligned[i] or ema34_1d_aligned[i] > ema34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals