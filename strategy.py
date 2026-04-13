#!/usr/bin/env python3
"""
12h_1D_1W_Camarilla_Pivot_Breakout_Volume_Confirmation_v1
Hypothesis: On 12h timeframe, breakouts above daily Camarilla H3 or below daily L3 with volume > 1.8x 50-period average and price aligned with weekly EMA200 trend yield reliable moves. Uses 1d for pivot/volume filters and 1w for trend filter. Designed for low-frequency, high-quality signals that work in both bull and bear markets by capturing institutional breakout attempts with volume confirmation and trend alignment. Target: 15-25 trades/year.
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
    
    # Volume confirmation: current volume > 1.8x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean()
    volume_expansion = volume > (vol_ma_50 * 1.8)
    
    # Daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_high_1d = df_1d['high'].values
    prev_low_1d = df_1d['low'].values
    prev_close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels (H3/L3 for breakout)
    range_1d = prev_high_1d - prev_low_1d
    camarilla_h3_1d = prev_close_1d + 1.1 * range_1d / 4
    camarilla_l3_1d = prev_close_1d - 1.1 * range_1d / 4
    
    # Align daily levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    # Weekly EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    prev_close_1w = df_1w['close'].values
    ema200_1w_raw = pd.Series(prev_close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w_raw)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    bars_since_entry = 0  # Track holding period
    
    for i in range(100, n):  # warmup period
        # Skip if any required data is not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_expansion[i]) or np.isnan(ema200_1w_aligned[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        # Long signal: break above daily Camarilla H3 with volume expansion and price above weekly EMA200
        long_signal = (close[i] > camarilla_h3_aligned[i] and 
                      volume_expansion[i] and 
                      close[i] > ema200_1w_aligned[i])
        
        # Short signal: break below daily Camarilla L3 with volume expansion and price below weekly EMA200
        short_signal = (close[i] < camarilla_l3_aligned[i] and 
                       volume_expansion[i] and 
                       close[i] < ema200_1w_aligned[i])
        
        # Exit conditions: minimum holding period (4 bars = 48 hours) reached and opposite signal
        if position == 1 and bars_since_entry >= 4 and short_signal:
            position = -1
            signals[i] = -position_size
            bars_since_entry = 0
        elif position == -1 and bars_since_entry >= 4 and long_signal:
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

name = "12h_1D_1W_Camarilla_Pivot_Breakout_Volume_Confirmation_v1"
timeframe = "12h"
leverage = 1.0