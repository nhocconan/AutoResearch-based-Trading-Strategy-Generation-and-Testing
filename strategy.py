#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + 1d EMA50 trend filter with volume spike confirmation.
Long when Bull Power > 0 AND close > 1d EMA50 AND volume > 2.0x 20-period average.
Short when Bear Power < 0 AND close < 1d EMA50 AND volume > 2.0x 20-period average.
Exit when power crosses zero (Bull Power <= 0 for long exit, Bear Power >= 0 for short exit).
Elder Ray measures bull/bear strength relative to EMA13, providing adaptive trend/mean-reversion edge.
1d EMA50 offers smooth HTF trend filter. Volume confirmation ensures institutional participation.
Designed for 6h timeframe to capture swing moves with controlled trade frequency (target: 12-37/year).
Works in bull markets via Bull Power strength and in bear markets via Bear Power mean-reversion exits.
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
    
    # Load 1d data for EMA50 - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Elder Ray components on 6h timeframe
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Align 1d EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50)  # Ensure warmup for EMA50 and EMA13
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND close > 1d EMA50 AND volume spike
            if (bull_power[i] > 0 and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND close < 1d EMA50 AND volume spike
            elif (bear_power[i] < 0 and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: power crosses zero (loss of bull/bear strength)
            if position == 1 and bull_power[i] <= 0:
                exit_signal = True
            elif position == -1 and bear_power[i] >= 0:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0