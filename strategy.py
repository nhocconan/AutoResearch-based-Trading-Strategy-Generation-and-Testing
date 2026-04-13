#!/usr/bin/env python3
"""
12h_1w_1d_Camarilla_Pivot_Breakout_With_Volume_Confirmation
Hypothesis: Trade 12h breakouts above weekly and daily Camarilla pivot levels with volume confirmation to capture strong momentum moves in both bull and bear markets. Weekly trend filter (price > weekly EMA50) ensures alignment with higher timeframe direction, reducing false breakouts. Volume > 1.5x 50-period average confirms institutional participation. Designed for low trade frequency (target 12-37/year) to minimize fee drag while maintaining edge in trending and ranging markets.
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
    
    # Weekly data for trend filter and higher timeframe context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Weekly EMA50 trend filter
    ema50_1w_raw = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean()
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w_raw)
    
    # Previous day's high/low/close for Camarilla calculation
    prev_high_1d = df_1d['high'].values
    prev_low_1d = df_1d['low'].values
    prev_close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels (H4 and L4 for breakout)
    camarilla_h4_1d = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_l4_1d = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2
    
    # Align daily Camarilla levels to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    bars_since_entry = 0  # Track holding period
    
    for i in range(60, n):  # warmup period
        # Skip if any required data is not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        # Long signal: break above daily Camarilla H4 with volume expansion and price above weekly EMA50
        long_signal = (close[i] > camarilla_h4_aligned[i] and 
                      volume_expansion[i] and 
                      close[i] > ema50_1w_aligned[i])
        
        # Short signal: break below daily Camarilla L4 with volume expansion and price below weekly EMA50
        short_signal = (close[i] < camarilla_l4_aligned[i] and 
                       volume_expansion[i] and 
                       close[i] < ema50_1w_aligned[i])
        
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

name = "12h_1w_1d_Camarilla_Pivot_Breakout_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0