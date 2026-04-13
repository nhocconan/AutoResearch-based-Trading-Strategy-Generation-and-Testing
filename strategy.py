#!/usr/bin/env python3
"""
1d_1W_Camarilla_Pivot_Breakout_With_Volume_Confirmation
Hypothesis: Buy when price breaks above weekly Camarilla H4 level with volume > 2x 20-period average, sell when price breaks below weekly L4 level with volume confirmation. Uses 1d primary timeframe with weekly trend filter via EMA50. Designed for low-frequency, high-conviction trades to minimize fee drag and work in both bull and bear markets by capturing genuine breakouts with strong volume and trend alignment.
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
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 2.0)
    
    # Previous week's high/low/close for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    prev_high_1w = df_1w['high'].values
    prev_low_1w = df_1w['low'].values
    prev_close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels
    camarilla_h4_1w = prev_close_1w + 1.1 * (prev_high_1w - prev_low_1w) / 2
    camarilla_l4_1w = prev_close_1w - 1.1 * (prev_high_1w - prev_low_1w) / 2
    
    # Align weekly levels to daily timeframe (wait for weekly close)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4_1w)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4_1w)
    
    # Weekly EMA50 trend filter
    ema50_1w_raw = pd.Series(prev_close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w_raw)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    bars_since_entry = 0  # Track holding period
    
    for i in range(50, n):  # warmup period
        # Skip if any required data is not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        # Long signal: break above weekly Camarilla H4 with volume expansion and price above weekly EMA50
        long_signal = (close[i] > camarilla_h4_aligned[i] and 
                      volume_expansion[i] and 
                      close[i] > ema50_1w_aligned[i])
        
        # Short signal: break below weekly Camarilla L4 with volume expansion and price below weekly EMA50
        short_signal = (close[i] < camarilla_l4_aligned[i] and 
                       volume_expansion[i] and 
                       close[i] < ema50_1w_aligned[i])
        
        # Exit conditions: minimum holding period reached and opposite signal
        if position == 1 and bars_since_entry >= 5 and short_signal:
            position = -1
            signals[i] = -position_size
            bars_since_entry = 0
        elif position == -1 and bars_since_entry >= 5 and long_signal:
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

name = "1d_1W_Camarilla_Pivot_Breakout_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0