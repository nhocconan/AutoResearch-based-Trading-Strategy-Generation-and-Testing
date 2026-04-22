#!/usr/bin/env python3

"""
Hypothesis: 12-hour Williams %R reversal with 1-day EMA34 trend filter and volume confirmation.
Trade reversals when Williams %R reaches extreme levels (>80 for short, <20 for long) 
but only in the direction of the daily EMA34 trend. Uses volume spike to confirm momentum.
Designed for low trade frequency (12-37/year) by requiring extreme %R levels, trend alignment,
and volume confirmation. Works in both bull and bear markets by following the daily trend.
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
    
    # Williams %R on 12h: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Values: 0 to -100, where > -20 is overbought, < -80 is oversold
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Load 1d data for EMA34 trend - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 12h
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + price above daily EMA34 + volume spike
            if williams_r[i] < -80 and close[i] > ema34_1d_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) + price below daily EMA34 + volume spike
            elif williams_r[i] > -20 and close[i] < ema34_1d_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral range (-50) or opposite extreme
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R rises above -50 or becomes overbought
                if williams_r[i] > -50 or williams_r[i] > -20:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R falls below -50 or becomes oversold
                if williams_r[i] < -50 or williams_r[i] < -80:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR_Reversal_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0