#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian channel breakout with 1-day trend filter and volume confirmation.
Long when price breaks above 20-period Donchian upper band and 1-day EMA50 is rising.
Short when price breaks below 20-period Donchian lower band and 1-day EMA50 is falling.
Exit when price returns to the Donchian midpoint or 1-day EMA50 reverses.
Designed for low trade frequency by requiring breakout + trend + volume confirmation.
Works in both bull and bear markets by following the 1-day trend.
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
    
    # Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (high_20 + low_20) / 2.0
    
    # Load 1-day data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 50-period EMA on 1-day close for trend
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(donch_mid[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: break above upper band + 1-day EMA50 rising + volume spike
            if close[i] > high_20[i] and ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band + 1-day EMA50 falling + volume spike
            elif close[i] < low_20[i] and ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: return to midpoint or 1-day EMA50 reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to midpoint or EMA turns down
                if close[i] <= donch_mid[i] or ema50_1d_aligned[i] < ema50_1d_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to midpoint or EMA turns up
                if close[i] >= donch_mid[i] or ema50_1d_aligned[i] > ema50_1d_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_DonchianBreakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0