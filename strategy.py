#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_supertrend_1d_hlc3_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HLC3 and Supertrend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate HLC3 for 1d
    hlc3 = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    
    # Supertrend on 1d: ATR(10) * 3
    atr_period = 10
    multiplier = 3
    
    # Calculate ATR
    tr1 = df_1d['high'].values - df_1d['low'].values
    tr2 = np.abs(df_1d['high'].values - df_1d['close'].shift(1).values)
    tr3 = np.abs(df_1d['low'].values - df_1d['close'].shift(1).values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Basic upper and lower bands
    basic_ub = (df_1d['high'].values + df_1d['low'].values) / 2 + multiplier * atr
    basic_lb = (df_1d['high'].values + df_1d['low'].values) / 2 - multiplier * atr
    
    # Initialize Supertrend components
    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()
    supertrend = np.zeros(len(hlc3))
    direction = np.ones(len(hlc3))  # 1 for uptrend, -1 for downtrend
    
    # Calculate Supertrend
    for i in range(1, len(hlc3)):
        if hlc3[i-1] > final_ub[i-1]:
            direction[i] = -1
        elif hlc3[i-1] < final_lb[i-1]:
            direction[i] = 1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            final_ub[i] = basic_ub[i]
            final_lb[i] = max(final_lb[i], final_lb[i-1])
            supertrend[i] = final_lb[i]
        else:
            final_ub[i] = min(final_ub[i], final_ub[i-1])
            final_lb[i] = basic_lb[i]
            supertrend[i] = final_ub[i]
    
    # Align Supertrend direction and value to 6h
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    
    # Volume confirmation: volume > 1.5x average of last 12 periods (12*6h = 3 days)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(direction_aligned[i]) or np.isnan(supertrend_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Supertrend or trend changes to down
            if close[i] < supertrend_aligned[i] or direction_aligned[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Supertrend or trend changes to up
            if close[i] > supertrend_aligned[i] or direction_aligned[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price above Supertrend, uptrend, volume confirmation
            if (close[i] > supertrend_aligned[i] and 
                direction_aligned[i] == 1 and 
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price below Supertrend, downtrend, volume confirmation
            elif (close[i] < supertrend_aligned[i] and 
                  direction_aligned[i] == -1 and 
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals