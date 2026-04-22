#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian breakout with 1-day EMA21 trend and volume spike.
Long when price breaks above 20-bar high with rising 1-day EMA21 and volume spike.
Short when price breaks below 20-bar low with falling 1-day EMA21 and volume spike.
Exit when price crosses 10-bar EMA on 12h chart.
Designed for low trade frequency by requiring multiple confirmations and using 12h-level structure.
Works in both bull and bear markets by following the daily trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for EMA trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1-day EMA21 for trend filter
    ema21_1d = pd.Series(df_1d['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    # 12-hour Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12-hour EMA10 for exit
    ema10_12h = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema21_1d_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(ema10_12h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above 20-bar high with rising 1-day EMA21 and volume spike
            if (close[i] > high_20[i] and 
                ema21_1d_aligned[i] > ema21_1d_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-bar low with falling 1-day EMA21 and volume spike
            elif (close[i] < low_20[i] and 
                  ema21_1d_aligned[i] < ema21_1d_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses 10-bar EMA on 12h chart
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below EMA10
                if close[i] < ema10_12h[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above EMA10
                if close[i] > ema10_12h[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian_Breakout_1dEMA21_Trend_Volume"
timeframe = "12h"
leverage = 1.0