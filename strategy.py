#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ElderRay_BullBearPower"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # === Daily EMA13 for Elder Ray (standard period) ===
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA13 manually to avoid pandas overhead
    ema13 = np.zeros_like(close_1d)
    ema13[0] = close_1d[0]
    alpha = 2.0 / (13 + 1)
    for i in range(1, len(close_1d)):
        ema13[i] = alpha * close_1d[i] + (1 - alpha) * ema13[i-1]
    
    # Elder Ray components
    bull_power = high_1d - ema13   # Bull Power = High - EMA13
    bear_power = low_1d - ema13    # Bear Power = Low - EMA13
    
    # Align to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13)
    
    # === 6h EMA13 for trend filter (avoid counter-trend trades) ===
    close_6h = prices['close'].values
    ema13_6h = np.zeros_like(close_6h)
    ema13_6h[0] = close_6h[0]
    for i in range(1, len(close_6h)):
        ema13_6h[i] = alpha * close_6h[i] + (1 - alpha) * ema13_6h[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start after EMA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        ema13_6h_val = ema13_6h[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema13_6h_val) or np.isnan(bull_val) or 
            np.isnan(bear_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Strong bull power AND price above EMA13 (uptrend)
            if bull_val > 0 and close_val > ema13_6h_val:
                signals[i] = 0.25
                position = 1
            # Short: Strong bear power AND price below EMA13 (downtrend)
            elif bear_val < 0 and close_val < ema13_6h_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bear power turns negative (selling pressure)
            if bear_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bull power turns positive (buying pressure)
            if bull_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals