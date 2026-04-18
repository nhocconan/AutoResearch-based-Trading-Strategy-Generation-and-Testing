#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h trend following using 1d Supertrend for direction, with volume confirmation and ATR-based exit.
# Long when price closes above Supertrend with volume > 1.5x 20-period average.
# Short when price closes below Supertrend with same volume condition.
# Exit when price crosses back below/above Supertrend.
# Uses 1d Supertrend for trend filter, volume surge for conviction, ATR stop for risk.
# Designed for ~20-40 trades/year per symbol.
name = "4h_1dSupertrend_VolumeSurge_ATR_Exit"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Supertrend calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Supertrend on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ATR(10) for Supertrend
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d)
    tr3 = np.abs(low_1d - close_1d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend parameters
    atr_multiplier = 3.0
    basic_ub = (high_1d + low_1d) / 2 + atr_multiplier * atr
    basic_lb = (high_1d + low_1d) / 2 - atr_multiplier * atr
    
    # Initialize final bands
    final_ub = np.full_like(close_1d, np.nan)
    final_lb = np.full_like(close_1d, np.nan)
    supertrend = np.full_like(close_1d, np.nan)
    trend = np.full_like(close_1d, 1)  # 1 for uptrend, -1 for downtrend
    
    # Calculate Supertrend
    for i in range(10, len(close_1d)):
        if i == 10:
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
        else:
            final_ub[i] = basic_ub[i] if (basic_ub[i] < final_ub[i-1] or close_1d[i-1] > final_ub[i-1]) else final_ub[i-1]
            final_lb[i] = basic_lb[i] if (basic_lb[i] > final_lb[i-1] or close_1d[i-1] < final_lb[i-1]) else final_lb[i-1]
        
        if i == 10:
            supertrend[i] = final_ub[i]
            trend[i] = 1
        else:
            if supertrend[i-1] == final_ub[i-1]:
                supertrend[i] = final_lb[i] if close_1d[i] <= final_lb[i] else final_ub[i]
                trend[i] = -1 if supertrend[i] == final_lb[i] else 1
            else:
                supertrend[i] = final_ub[i] if close_1d[i] >= final_ub[i] else final_lb[i]
                trend[i] = 1 if supertrend[i] == final_ub[i] else -1
    
    # Align Supertrend to 4h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    
    # Volume filter: current volume > 1.5 * 20-period average (20 * 4h = ~3.33 days)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(supertrend_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        supertrend_val = supertrend_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: price above Supertrend with volume surge
            if close_val > supertrend_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below Supertrend with volume surge
            elif close_val < supertrend_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below Supertrend
            if close_val < supertrend_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above Supertrend
            if close_val > supertrend_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals