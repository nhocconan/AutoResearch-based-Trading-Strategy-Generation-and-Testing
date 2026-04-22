#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian channel breakout with 1-day trend filter and volume confirmation.
Long when price breaks above 20-period Donchian high, 1-day EMA50 is rising, and volume spike.
Short when price breaks below 20-period Donchian low, 1-day EMA50 is falling, and volume spike.
Exit when price crosses 10-period Donchian midline or trend reverses.
Designed for low trade frequency (<50/year) by requiring price structure + trend + volume.
Works in both bull and bear markets by following the 1-day trend.
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
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 50-period EMA on 1d close for trend
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: break above Donchian high, 1d EMA50 rising, volume spike
            if (close[i] > highest_high[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low, 1d EMA50 falling, volume spike
            elif (close[i] < lowest_low[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: cross midline or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: cross below midline or 1d EMA50 turns down
                if close[i] < donchian_mid[i] or ema50_1d_aligned[i] < ema50_1d_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: cross above midline or 1d EMA50 turns up
                if close[i] > donchian_mid[i] or ema50_1d_aligned[i] > ema50_1d_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0