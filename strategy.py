#!/usr/bin/env python3
"""
12h_1W_1D_Camarilla_Pivot_Breakout_With_Volume_Filter
Hypothesis: Buy when price breaks above weekly Camarilla H3 level with volume > 1.5x 50-period average, sell when price breaks below weekly L3 level. Uses 12h primary timeframe with weekly trend filter via price > weekly EMA50. Designed to capture major trend breaks in both bull and bear markets with confirmation from volume and weekly trend alignment. Low trade frequency target: 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: current volume > 1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean()
    volume_expansion = volume > (vol_ma_50 * 1.5)
    
    # Weekly data for Camarilla calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    prev_high_1w = df_1w['high'].values
    prev_low_1w = df_1w['low'].values
    prev_close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels (H3 and L3)
    camarilla_h3_1w = prev_close_1w + 1.1 * (prev_high_1w - prev_low_1w) / 4
    camarilla_l3_1w = prev_close_1w - 1.1 * (prev_high_1w - prev_low_1w) / 4
    
    # Weekly EMA50 trend filter
    ema50_1w_raw = pd.Series(prev_close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w_raw)
    
    # Align weekly levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3_1w)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    bars_since_entry = 0  # Track holding period
    
    for i in range(60, n):  # warmup period
        # Skip if any required data is not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        # Long signal: break above weekly Camarilla H3 with volume expansion and price above weekly EMA50
        long_signal = (close[i] > camarilla_h3_aligned[i] and 
                      volume_expansion[i] and 
                      close[i] > ema50_1w_aligned[i])
        
        # Short signal: break below weekly Camarilla L3 with volume expansion and price below weekly EMA50
        short_signal = (close[i] < camarilla_l3_aligned[i] and 
                       volume_expansion[i] and 
                       close[i] < ema50_1w_aligned[i])
        
        # Exit conditions: minimum holding period reached and opposite signal
        if position == 1 and bars_since_entry >= 8 and short_signal:
            position = -1
            signals[i] = -position_size
            bars_since_entry = 0
        elif position == -1 and bars_since_entry >= 8 and long_signal:
            position = 1
            signals[i] = position_size
            bars_since_entry = 0
        elif position == 0:
            if long_signal:
                position = 1
                signals[i] = position_size
                bars_since_entry = 0
            elif short_signal:
                position = -1
                signals[i] = -position_size
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "12h_1W_1D_Camarilla_Pivot_Breakout_With_Volume_Filter"
timeframe = "12h"
leverage = 1.0