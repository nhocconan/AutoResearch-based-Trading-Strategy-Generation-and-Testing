#!/usr/bin/env python3
"""
4h_1D_Camarilla_Pivot_Breakout_Trend_HTF_v1
Hypothesis: Buy when price breaks above daily Camarilla H4 level with volume > 2x 50-period average and price > weekly EMA50, sell when price breaks below daily L4 level with volume confirmation and price < weekly EMA50. Uses 4h primary timeframe with 1w trend filter to reduce false breakouts in choppy markets. Weekly trend filter adds robustness in both bull and bear markets by ensuring trades align with higher-timeframe momentum. Volume confirmation ensures breakout strength. Designed for low trade frequency (<30/year) to minimize fee drag.
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
    
    # Volume confirmation: current volume > 2x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean()
    volume_expansion = volume > (vol_ma_50 * 2.0)
    
    # Previous day's high/low/close for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_high_1d = df_1d['high'].values
    prev_low_1d = df_1d['low'].values
    prev_close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels
    camarilla_h4_1d = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_l4_1d = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2
    
    # Align daily levels to 4h timeframe (wait for daily close)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    
    # Weekly EMA50 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    prev_close_1w = df_1w['close'].values
    ema50_1w_raw = pd.Series(prev_close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w_raw)
    
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
        
        # Long signal: break above daily Camarilla H4 with volume expansion and price > weekly EMA50
        long_signal = (close[i] > camarilla_h4_aligned[i] and 
                      volume_expansion[i] and 
                      close[i] > ema50_1w_aligned[i])
        
        # Short signal: break below daily Camarilla L4 with volume expansion and price < weekly EMA50
        short_signal = (close[i] < camarilla_l4_aligned[i] and 
                       volume_expansion[i] and 
                       close[i] < ema50_1w_aligned[i])
        
        # Exit conditions: minimum holding period reached and opposite signal
        if position == 1 and bars_since_entry >= 12 and short_signal:
            position = -1
            signals[i] = -position_size
            bars_since_entry = 0
        elif position == -1 and bars_since_entry >= 12 and long_signal:
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

name = "4h_1D_Camarilla_Pivot_Breakout_Trend_HTF_v1"
timeframe = "4h"
leverage = 1.0