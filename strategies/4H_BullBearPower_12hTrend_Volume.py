#!/usr/bin/env python3
"""
Hypothesis: 4-hour Bull/Bear Power with 12-hour Trend and Volume Confirmation.
Long when Bull Power > 0, Bear Power < 0, and 12h EMA50 is rising with volume spike.
Short when Bear Power < 0, Bull Power > 0, and 12h EMA50 is falling with volume spike.
Exit when Bull/Bear Power crosses zero or 12h EMA50 reverses.
Designed for low trade frequency by requiring multiple confirmations.
Works in both bull and bear markets by following the 12h trend.
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
    
    # Bull Power = High - EMA13(Close)
    # Bear Power = Low - EMA13(Close)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 50-period EMA on 12h close for trend
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, 12h EMA50 rising, volume spike
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, Bull Power > 0, 12h EMA50 falling, volume spike
            elif (bear_power[i] < 0 and bull_power[i] > 0 and 
                  ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Power crosses zero or 12h EMA50 reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Bull Power <= 0 or 12h EMA50 turns down
                if bull_power[i] <= 0 or ema50_12h_aligned[i] < ema50_12h_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Bear Power >= 0 or 12h EMA50 turns up
                if bear_power[i] >= 0 or ema50_12h_aligned[i] > ema50_12h_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_BullBearPower_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0