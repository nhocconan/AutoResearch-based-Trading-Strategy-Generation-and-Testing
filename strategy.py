#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian Channel (20) breakout with 1-day EMA trend filter and volume confirmation.
Long when price breaks above upper band, 1-day EMA50 is rising, and volume exceeds 1.5x 20-period average.
Short when price breaks below lower band, 1-day EMA50 is falling, and volume exceeds 1.5x 20-period average.
Exit when price crosses the opposite Donchian band or EMA trend reverses.
Designed for low trade frequency by requiring multiple confirmations and using 12h timeframe.
Works in both bull and bear markets by following daily trend while using 12h Donchian breakouts for entries.
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
    
    # Load 1-day data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian Channel (20 periods) - using lookback window (exclude current bar)
    high_20 = np.full(n, np.nan)
    low_20 = np.full(n, np.nan)
    for i in range(20, n):
        high_20[i] = np.max(high[i-20:i])  # Lookback 20 periods, exclude current
        low_20[i] = np.min(low[i-20:i])
    
    # Volume confirmation: current volume > 1.5x 20-period average volume
    vol_ma20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_ratio = np.full(n, np.nan)
    for i in range(20, n):
        if vol_ma20[i] > 0:
            volume_ratio[i] = volume[i] / vol_ma20[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper band, 1-day EMA50 rising, volume confirmation
            if (close[i] > high_20[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band, 1-day EMA50 falling, volume confirmation
            elif (close[i] < low_20[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below lower band OR EMA trend turns down
                if (close[i] < low_20[i] or 
                    ema50_1d_aligned[i] < ema50_1d_aligned[i-1]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above upper band OR EMA trend turns up
                if (close[i] > high_20[i] or 
                    ema50_1d_aligned[i] > ema50_1d_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0