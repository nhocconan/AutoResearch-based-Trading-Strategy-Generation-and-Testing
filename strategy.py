#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian channel breakout with 1-day trend filter (EMA200) and volume confirmation.
Long when price breaks above Donchian upper channel (20-period) with EMA200 uptrend and volume > 1.5x average.
Short when price breaks below Donchian lower channel with EMA200 downtrend and volume > 1.5x average.
Exit when price returns to Donchian middle (mean) or reverses with volume confirmation.
Designed for low trade frequency (~20-50/year) to capture strong trends while minimizing whipsaws and fee impact.
Works in both bull and bear markets by requiring trend alignment (EMA200 slope).
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
    
    # Donchian channel (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle = (highest_high + lowest_low) / 2.0
    
    # Load 1-day data for EMA200 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1-day EMA200
    close_1d = df_1d['close'].values
    ema200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align EMA200 to 4h timeframe
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200)
    
    # Volume average (20-period) on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema200_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema200_val = ema200_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper, EMA200 uptrend, volume confirmation
            if (close[i] > highest_high[i] and 
                close[i] > ema200_val and  # Price above EMA200 (uptrend)
                vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower, EMA200 downtrend, volume confirmation
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema200_val and  # Price below EMA200 (downtrend)
                  vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to middle OR breaks below lower with volume
                if (close[i] >= middle[i] or 
                    (close[i] < lowest_low[i] and vol_current > vol_ma_val)):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to middle OR breaks above upper with volume
                if (close[i] <= middle[i] or 
                    (close[i] > highest_high[i] and vol_current > vol_ma_val)):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dEMA200_Volume_Trend"
timeframe = "4h"
leverage = 1.0