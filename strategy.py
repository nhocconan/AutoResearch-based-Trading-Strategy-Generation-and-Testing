#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian channel breakout (20-period) with 1-day EMA trend filter and volume confirmation.
Long when price breaks above upper Donchian band, price > EMA34, and volume > 1.5x average.
Short when price breaks below lower Donchian band, price < EMA34, and volume > 1.5x average.
Exit when price returns to middle Donchian band or EMA trend reverses.
Designed for low trade frequency (~20-40/year) to capture strong trends while minimizing whipsaws.
Works in both bull and bear markets by requiring EMA trend alignment and volume confirmation.
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
    
    # Load 1-day data for EMA - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA (34-period)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle_band = (highest_high + lowest_low) / 2
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema_34_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian, above EMA, volume confirmation
            if (close[i] > highest_high[i] and close[i] > ema_val and 
                vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian, below EMA, volume confirmation
            elif (close[i] < lowest_low[i] and close[i] < ema_val and 
                  vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to middle band OR price crosses below EMA
                if close[i] <= middle_band[i] or close[i] < ema_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to middle band OR price crosses above EMA
                if close[i] >= middle_band[i] or close[i] > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_1dEMA34_Volume_Breakout"
timeframe = "4h"
leverage = 1.0