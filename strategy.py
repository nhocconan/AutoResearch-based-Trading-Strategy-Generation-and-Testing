#!/usr/bin/env python3
"""
6h_Rayleigh_Trend_Filter
Hypothesis: Use Elder Ray (Bull/Bear Power) with 12h EMA13 trend filter and volume confirmation on 6h timeframe.
Elder Ray measures bull/bear power relative to EMA13. In bull markets, buy when bull power > 0 and rising.
In bear markets, sell when bear power < 0 and falling. Volume surge confirms institutional participation.
Targets 15-30 trades/year to minimize fee drag. Works in both bull and bear via trend-adaptive logic.
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
    
    # Calculate EMA13 for Elder Ray on 6h data
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate 12h EMA13 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    ema13_12h = pd.Series(df_12h['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_12h_aligned = align_htf_to_ltf(prices, df_12h, ema13_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema13_12h_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA13
        trend_up = close[i] > ema13_12h_aligned[i]
        trend_down = close[i] < ema13_12h_aligned[i]
        
        # Elder Ray conditions with slope (using 3-bar change)
        bull_power_rising = bull_power[i] > bull_power[i-3]
        bear_power_falling = bear_power[i] < bear_power[i-3]
        
        # Entry conditions
        # Long: bull power > 0 AND rising AND uptrend + volume surge
        long_entry = (bull_power[i] > 0) and bull_power_rising and trend_up and volume_surge[i]
        # Short: bear power < 0 AND falling AND downtrend + volume surge
        short_entry = (bear_power[i] < 0) and bear_power_falling and trend_down and volume_surge[i]
        
        # Exit conditions: opposite Elder Ray signal or trend reversal
        long_exit = (bear_power[i] < 0) or not trend_up
        short_exit = (bull_power[i] > 0) or not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Rayleigh_Trend_Filter"
timeframe = "6h"
leverage = 1.0