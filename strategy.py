#!/usr/bin/env python3
# 12h_W1_Donchian_Breakout_Volume_Cam1D
# Hypothesis: On 12h chart, enter long when price breaks above weekly Donchian upper band with volume confirmation,
# enter short when price breaks below weekly Donchian lower band with volume confirmation.
# Filter: Price must be above/below daily 34-period EMA (trend filter).
# Exit: Price closes back inside Donchian channel.
# Uses weekly structure for trend direction, daily EMA for trend filter, 12h for execution.
# Designed for low trade frequency (~15-25/year) to minimize fee decay and work in trending markets.
# Weekly Donchian adapts to volatility, reducing false breakouts.
# Works in both bull and bear markets by capturing breakouts with volume and trend filters.

timeframe = "12h"
name = "12h_W1_Donchian_Breakout_Volume_Cam1D"
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
    
    # Weekly Donchian Channel (20-period)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    donch_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    # Daily EMA34 (trend filter)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after warmup for Donchian
        # Skip if any critical value is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian upper band + volume + above daily EMA34
            if close[i] > donch_high_aligned[i] and volume[i] > 1.5 * vol_ma[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian lower band + volume + below daily EMA34
            elif close[i] < donch_low_aligned[i] and volume[i] > 1.5 * vol_ma[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price closes back inside weekly Donchian channel
            if close[i] < donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes back inside weekly Donchian channel
            if close[i] > donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals