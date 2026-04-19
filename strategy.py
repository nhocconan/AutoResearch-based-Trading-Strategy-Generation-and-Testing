#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Supertrend_Filter_Volume_Spike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Supertrend parameters
    atr_period = 10
    atr_multiplier = 3.0
    
    # Calculate ATR
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Basic upper and lower bands
    basic_ub = (high_1d + low_1d) / 2 + atr_multiplier * atr
    basic_lb = (high_1d + low_1d) / 2 - atr_multiplier * atr
    
    # Initialize Supertrend arrays
    final_ub = np.full_like(close_1d, np.nan)
    final_lb = np.full_like(close_1d, np.nan)
    supertrend = np.full_like(close_1d, np.nan)
    direction = np.full_like(close_1d, np.nan)  # 1 for up, -1 for down
    
    # Calculate Supertrend
    for i in range(1, len(close_1d)):
        if np.isnan(atr[i]) or np.isnan(basic_ub[i]) or np.isnan(basic_lb[i]):
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
            continue
            
        # Upper band logic
        if i == 1:
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
        else:
            if basic_ub[i] < final_ub[i-1] or close_1d[i-1] > final_ub[i-1]:
                final_ub[i] = basic_ub[i]
            else:
                final_ub[i] = final_ub[i-1]
                
            # Lower band logic
            if basic_lb[i] > final_lb[i-1] or close_1d[i-1] < final_lb[i-1]:
                final_lb[i] = basic_lb[i]
            else:
                final_lb[i] = final_lb[i-1]
        
        # Supertrend logic
        if i == 1:
            supertrend[i] = final_lb[i]
            direction[i] = 1
        else:
            if supertrend[i-1] == final_ub[i-1]:
                if close_1d[i] <= final_ub[i]:
                    supertrend[i] = final_ub[i]
                    direction[i] = -1
                else:
                    supertrend[i] = final_ub[i]
                    direction[i] = 1
            else:
                if close_1d[i] >= final_lb[i]:
                    supertrend[i] = final_lb[i]
                    direction[i] = 1
                else:
                    supertrend[i] = final_lb[i]
                    direction[i] = -1
    
    # Align Supertrend direction to 12h timeframe
    supertrend_dir_12h = align_htf_to_ltf(prices, df_1d, direction)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(supertrend_dir_12h[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 2.0x average
        volume_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long: Supertrend up + volume spike
            if supertrend_dir_12h[i] == 1 and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Supertrend down + volume spike
            elif supertrend_dir_12h[i] == -1 and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Supertrend flips down
            if supertrend_dir_12h[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Supertrend flips up
            if supertrend_dir_12h[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals