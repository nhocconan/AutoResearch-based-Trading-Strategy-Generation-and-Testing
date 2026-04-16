#!/usr/bin/env python3
"""
6h_ElderRay_EMA_Trend
Hypothesis: Elder Ray Index (bull/bear power) combined with EMA trend filter captures
strong momentum moves while avoiding whipsaws. Works in both bull and bear markets
by requiring alignment between Elder Ray signal and EMA direction.
Uses 6h for execution, 12h for EMA trend filter to reduce noise.
Target: 50-150 trades over 4 years (12-37/year) with disciplined entries.
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
    
    # === 6h data (primary) ===
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # === 12h data (HTF for EMA trend filter) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # === Elder Ray Index (13-period) ===
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema13 = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_6h - ema13
    bear_power = low_6h - ema13
    
    # === EMA trend filter (21-period on 12h) ===
    ema21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align all HTF data to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    
    signals = np.zeros(n)
    
    # Warmup: enough for EMA calculations
    warmup = 30
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema21_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        bull = bull_power_aligned[i]
        bear = bear_power_aligned[i]
        ema21 = ema21_12h_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: Bear power becomes positive (selling pressure) OR price closes below EMA
            if bear > 0 or price < ema21:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: Bull power becomes negative (buying pressure) OR price closes above EMA
            if bull < 0 or price > ema21:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bull power positive (buying pressure) AND price above EMA (uptrend)
            if bull > 0 and price > ema21:
                signals[i] = 0.25
                position = 1
                continue
            # SHORT: Bear power negative (selling pressure) AND price below EMA (downtrend)
            elif bear < 0 and price < ema21:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_EMA_Trend"
timeframe = "6h"
leverage = 1.0