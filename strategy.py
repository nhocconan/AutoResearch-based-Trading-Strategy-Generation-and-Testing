#!/usr/bin/env python3
"""
Hypothesis: 1-hour trend following using 4-hour Supertrend for direction and 1-day volume spike for confirmation.
Long when price > Supertrend and volume > 1.5x average, short when price < Supertrend and volume > 1.5x average.
Exit when price crosses Supertrend in opposite direction.
Uses 4h for trend direction (fewer signals) and 1h for entry timing. Volume filter reduces false signals.
Designed for low trade frequency (~20-40/year) to avoid fee drag. Works in bull/bear by requiring volume confirmation.
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
    
    # Load 4-hour data for Supertrend - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Load 1-day data for volume average - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4-hour Supertrend (ATR=10, multiplier=3.0)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high_4h + low_4h) / 2 + 3.0 * atr
    basic_lb = (high_4h + low_4h) / 2 - 3.0 * atr
    
    # Final Upper and Lower Bands
    final_ub = np.zeros_like(basic_ub)
    final_lb = np.zeros_like(basic_lb)
    
    for i in range(len(close_4h)):
        if i == 0:
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
        else:
            if basic_ub[i] < final_ub[i-1] or close_4h[i-1] > final_ub[i-1]:
                final_ub[i] = basic_ub[i]
            else:
                final_ub[i] = final_ub[i-1]
                
            if basic_lb[i] > final_lb[i-1] or close_4h[i-1] < final_lb[i-1]:
                final_lb[i] = basic_lb[i]
            else:
                final_lb[i] = final_lb[i-1]
    
    # Supertrend
    supertrend = np.zeros_like(close_4h)
    for i in range(len(close_4h)):
        if i == 0:
            supertrend[i] = final_ub[i]
        else:
            if supertrend[i-1] == final_ub[i-1] and close_4h[i] <= final_ub[i]:
                supertrend[i] = final_ub[i]
            elif supertrend[i-1] == final_ub[i-1] and close_4h[i] > final_ub[i]:
                supertrend[i] = final_lb[i]
            elif supertrend[i-1] == final_lb[i-1] and close_4h[i] >= final_lb[i]:
                supertrend[i] = final_lb[i]
            elif supertrend[i-1] == final_lb[i-1] and close_4h[i] < final_lb[i]:
                supertrend[i] = final_ub[i]
    
    # Calculate 1-day volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to lower timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_4h, supertrend)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Session filter: 8-20 UTC (already datetime64 index)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup period
        # Skip if data not ready or outside session
        if (np.isnan(supertrend_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            not (8 <= hours[i] <= 20)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        st = supertrend_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price above Supertrend and volume spike
            if close[i] > st and vol_current > 1.5 * vol_ma:
                signals[i] = 0.20
                position = 1
            # Short: price below Supertrend and volume spike
            elif close[i] < st and vol_current > 1.5 * vol_ma:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below Supertrend
                if close[i] < st:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above Supertrend
                if close[i] > st:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Supertrend_4hDir_1dVolSpike"
timeframe = "1h"
leverage = 1.0