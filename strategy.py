#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d EMA21 trend filter
# Uses Donchian channel breakout as primary signal, confirmed by volume spike and daily trend.
# Designed to capture trends while avoiding whipsaws in ranging markets.
# Targets 20-50 trades per year (~80-200 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the higher timeframe trend.

name = "4h_Donchian20_1dEMA21_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA21 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema21_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper Donchian band, volume confirmation, 1d uptrend
            if close[i] > high_20[i] and vol_conf[i] and ema21_1d_aligned[i] > ema21_1d[0]:  # trend filter uses current vs first available
                signals[i] = 0.30
                position = 1
            # Enter short: price breaks below lower Donchian band, volume confirmation, 1d downtrend
            elif close[i] < low_20[i] and vol_conf[i] and ema21_1d_aligned[i] < ema21_1d[0]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian band or trend turns down
            if close[i] < low_20[i] or ema21_1d_aligned[i] < ema21_1d[0]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price breaks above upper Donchian band or trend turns up
            if close[i] > high_20[i] or ema21_1d_aligned[i] > ema21_1d[0]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals