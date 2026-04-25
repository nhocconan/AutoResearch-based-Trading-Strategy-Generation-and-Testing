#!/usr/bin/env python3
"""
6h_Camarilla_H3L3_Breakout_1dTrendFilter_v3
Hypothesis: Trade Camarilla H3/L3 breakouts on 6h with 1d EMA50 trend filter.
Only long when price breaks above H3 and 1d EMA50 rising; only short when price breaks below L3 and 1d EMA50 falling.
Add volume confirmation (volume > 1.3 * 20-bar median volume) to avoid false breakouts.
Target: 15-25 trades/year to minimize fee drag while capturing sustained moves.
Discrete sizing: 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous 6h bar
    # H3 = close + 1.1*(high-low)/2
    # L3 = close - 1.1*(high-low)/2
    # We need previous bar's high, low, close
    prev_high = np.concatenate([[high[0]], high[:-1]])
    prev_low = np.concatenate([[low[0]], low[:-1]])
    prev_close = np.concatenate([[close[0]], close[:-1]])
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Volume confirmation: volume > 1.3 * 20-bar median volume
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > 1.3 * vol_median
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    # Start index: need warmup for 1d EMA50 (50) and Camarilla (1) and volume median (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(vol_median[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Determine 1d EMA50 trend direction (rising/falling)
        if i > start_idx:
            ema_50_prev = ema_50_1d_aligned[i-1]
            ema_50_rising = ema_50_1d_aligned[i] > ema_50_prev
            ema_50_falling = ema_50_1d_aligned[i] < ema_50_prev
        else:
            ema_50_rising = False
            ema_50_falling = False
        
        if position == 0:
            # Long setup: price breaks above H3 AND 1d EMA50 rising AND volume confirmation
            long_setup = (close[i] > camarilla_h3[i]) and ema_50_rising and vol_confirm
            
            # Short setup: price breaks below L3 AND 1d EMA50 falling AND volume confirmation
            short_setup = (close[i] < camarilla_l3[i]) and ema_50_falling and vol_confirm
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            elif short_setup:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            bars_since_entry += 1
            # Exit: price re-enters Camarilla H3-L3 range OR 1d EMA50 turns falling OR max hold (24 bars = 4 days)
            if (close[i] < camarilla_h3[i]) or (not ema_50_rising) or (bars_since_entry >= 24):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            bars_since_entry += 1
            # Exit: price re-enters Camarilla H3-L3 range OR 1d EMA50 turns rising OR max hold (24 bars = 4 days)
            if (close[i] > camarilla_l3[i]) or (not ema_50_falling) or (bars_since_entry >= 24):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1dTrendFilter_v3"
timeframe = "6h"
leverage = 1.0