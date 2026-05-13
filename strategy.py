#!/usr/bin/env python3
# 12h_WheelersTrend_Momentum_Crossover
# Hypothesis: Use weekly EMA21 for trend direction and daily EMA13/EMA34 crossover for momentum entry. 
# Enter long when weekly EMA21 rising and daily EMA13 crosses above EMA34, confirmed by volume spike.
# Enter short when weekly EMA21 falling and daily EMA13 crosses below EMA34, confirmed by volume spike.
# Weekly trend filter reduces false signals in choppy markets. Momentum crossover captures swing points.
# Works in bull (long with weekly uptrend) and bear (short with weekly downtrend).
# Low frequency due to weekly trend requirement and crossover precision.

name = "12h_WheelersTrend_Momentum_Crossover"
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

    # Get weekly data for trend
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Get daily data for momentum
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly trend to 12h timeframe
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Align daily EMAs to 12h timeframe
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: volume > 1.5 * 2-period average (1 day worth at 12h)
    vol_ma_2 = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    volume_spike = volume > 1.5 * vol_ma_2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(ema21_1w_aligned[i]) or 
            np.isnan(ema13_1d_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Weekly uptrend + EMA13 crosses above EMA34 + volume spike
            if (ema21_1w_aligned[i] > ema21_1w_aligned[i-1] and 
                ema13_1d_aligned[i] > ema34_1d_aligned[i] and 
                ema13_1d_aligned[i-1] <= ema34_1d_aligned[i-1] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Weekly downtrend + EMA13 crosses below EMA34 + volume spike
            elif (ema21_1w_aligned[i] < ema21_1w_aligned[i-1] and 
                  ema13_1d_aligned[i] < ema34_1d_aligned[i] and 
                  ema13_1d_aligned[i-1] >= ema34_1d_aligned[i-1] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Weekly trend turns down OR EMA13 crosses below EMA34
            if (ema21_1w_aligned[i] < ema21_1w_aligned[i-1] or 
                (ema13_1d_aligned[i] < ema34_1d_aligned[i] and 
                 ema13_1d_aligned[i-1] >= ema34_1d_aligned[i-1])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Weekly trend turns up OR EMA13 crosses above EMA34
            if (ema21_1w_aligned[i] > ema21_1w_aligned[i-1] or 
                (ema13_1d_aligned[i] > ema34_1d_aligned[i] and 
                 ema13_1d_aligned[i-1] <= ema34_1d_aligned[i-1])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals