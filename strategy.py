#!/usr/bin/env python3
"""
Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation.
Long when price breaks above Donchian upper band, weekly trend is up, and volume spikes.
Short when price breaks below Donchian lower band, weekly trend is down, and volume spikes.
Exit when price returns to Donchian middle band or trend reverses.
Designed for low trade frequency by requiring breakout + trend + volume confirmation.
Works in both bull and bear markets by following the weekly trend.
"""

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
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2.0
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 34-period EMA on weekly close for trend
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above upper band, weekly trend up, volume spike
            if (close[i] > high_20[i] and ema34_1w_aligned[i] > ema34_1w_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band, weekly trend down, volume spike
            elif (close[i] < low_20[i] and ema34_1w_aligned[i] < ema34_1w_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to middle band or weekly trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below middle band or weekly trend turns down
                if close[i] < mid_20[i] or ema34_1w_aligned[i] < ema34_1w_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above middle band or weekly trend turns up
                if close[i] > mid_20[i] or ema34_1w_aligned[i] > ema34_1w_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "Daily_Donchian_20_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0