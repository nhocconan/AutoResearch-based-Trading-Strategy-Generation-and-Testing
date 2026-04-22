#!/usr/bin/env python3
"""
Hypothesis: 1-hour Fisher Transform with 4-hour trend and volume confirmation.
Long when Fisher crosses above -1.5 and 4h EMA50 rising with volume spike.
Short when Fisher crosses below +1.5 and 4h EMA50 falling with volume spike.
Exit when Fisher reverses or 4h EMA50 changes direction.
Uses Fisher Transform for early reversal signals in both bull and bear markets.
Targets 15-30 trades/year by requiring multiple confirmations.
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
    
    # Fisher Transform (9-period)
    hl2 = (high + low) / 2.0
    max_hl2 = pd.Series(hl2).rolling(window=9, min_periods=9).max().values
    min_hl2 = pd.Series(hl2).rolling(window=9, min_periods=9).min().values
    # Avoid division by zero
    diff = max_hl2 - min_hl2
    diff[diff == 0] = 1e-10
    value = 2.0 * ((hl2 - min_hl2) / diff - 0.5)
    value = np.clip(value, -0.999, 0.999)
    fish = 0.5 * np.log((1.0 + value) / (1.0 - value))
    # Smooth
    fish = pd.Series(fish).ewm(alpha=0.5, adjust=False).mean().values
    
    # Load 4h data for trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 50-period EMA on 4h close for trend
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(fish[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Fisher crosses above -1.5, 4h EMA50 rising, volume spike
            if (fish[i] > -1.5 and fish[i-1] <= -1.5 and 
                ema50_4h_aligned[i] > ema50_4h_aligned[i-1] and vol_spike):
                signals[i] = 0.20
                position = 1
            # Short: Fisher crosses below +1.5, 4h EMA50 falling, volume spike
            elif (fish[i] < 1.5 and fish[i-1] >= 1.5 and 
                  ema50_4h_aligned[i] < ema50_4h_aligned[i-1] and vol_spike):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: Fisher reverses or 4h EMA50 changes direction
            exit_signal = False
            
            if position == 1:
                # Exit long: Fisher crosses below +1.5 or 4h EMA50 turns down
                if fish[i] < 1.5 and fish[i-1] >= 1.5:
                    exit_signal = True
                elif ema50_4h_aligned[i] < ema50_4h_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Fisher crosses above -1.5 or 4h EMA50 turns up
                if fish[i] > -1.5 and fish[i-1] <= -1.5:
                    exit_signal = True
                elif ema50_4h_aligned[i] > ema50_4h_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Fisher_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0