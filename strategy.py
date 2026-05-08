#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d EMA trend filter and volume confirmation
# This strategy uses Donchian channel breakouts (20-period) on 12h timeframe,
# filtered by 1d EMA21 trend direction and volume spikes.
# Donchian breakouts capture breakout momentum, while EMA filter ensures
# alignment with higher timeframe trend to avoid counter-trend trades.
# Volume confirmation adds conviction to breakouts.
# Targets 12-30 trades per year (~48-120 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by only taking trades in direction of 1d trend.

name = "12h_Donchian20_1dEMA21_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period) on 12h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate EMA21 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 1d EMA21 to 12h timeframe
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    # Volume confirmation: current volume > 2.5x 20-period average (high threshold for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema21_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, 1d uptrend, volume confirmation
            if close[i] > donchian_high[i] and close[i] > ema21_1d_aligned[i] and vol_conf[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, 1d downtrend, volume confirmation
            elif close[i] < donchian_low[i] and close[i] < ema21_1d_aligned[i] and vol_conf[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low or 1d trend turns down
            if close[i] < donchian_low[i] or close[i] < ema21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high or 1d trend turns up
            if close[i] > donchian_high[i] or close[i] > ema21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals