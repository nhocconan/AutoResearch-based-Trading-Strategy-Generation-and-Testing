#!/usr/bin/env python3
"""
12h_1D_Camarilla_Breakout_Reversal_v1
Hypothesis: In range-bound markets, price often reverses from daily Camarilla H3/L3 levels.
Go short at H3 with volume confirmation and long at L3 with volume confirmation.
Uses 12h primary timeframe with 1d trend filter (price above/below daily EMA200 to filter counter-trend trades).
Designed to work in both bull and bear markets by fading extremes in ranges while respecting major trend.
Target: 15-30 trades/year (60-120 over 4 years) to minimize fee drag.
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
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    # Daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    prev_high_1d = df_1d['high'].values
    prev_low_1d = df_1d['low'].values
    prev_close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels (H3/L3 for reversals)
    range_1d = prev_high_1d - prev_low_1d
    camarilla_h3_1d = prev_close_1d + 1.1 * range_1d / 4
    camarilla_l3_1d = prev_close_1d - 1.1 * range_1d / 4
    
    # Daily EMA200 for trend filter
    ema200_1d_raw = pd.Series(prev_close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d_raw)
    
    # Align daily levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    bars_since_entry = 0  # Track holding period
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        # Short signal: price below daily Camarilla H3 with volume expansion and below daily EMA200 (fading rally in downtrend)
        short_signal = (close[i] < camarilla_h3_aligned[i] and 
                       volume_expansion[i] and 
                       close[i] < ema200_1d_aligned[i])
        
        # Long signal: price above daily Camarilla L3 with volume expansion and above daily EMA200 (fading drop in uptrend)
        long_signal = (close[i] > camarilla_l3_aligned[i] and 
                      volume_expansion[i] and 
                      close[i] > ema200_1d_aligned[i])
        
        # Exit conditions: minimum holding period (4 bars = 48 hours) or opposite signal
        if position == 1 and (bars_since_entry >= 4 or short_signal):
            position = -1 if short_signal else 0
            signals[i] = -position_size if short_signal else 0.0
            bars_since_entry = 0
        elif position == -1 and (bars_since_entry >= 4 or long_signal):
            position = 1 if long_signal else 0
            signals[i] = position_size if long_signal else 0.0
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

name = "12h_1D_Camarilla_Breakout_Reversal_v1"
timeframe = "12h"
leverage = 1.0