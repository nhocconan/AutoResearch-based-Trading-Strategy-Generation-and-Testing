#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for HTF analysis (Elder Ray)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 13-period EMA for Elder Ray calculation
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power and Bear Power
    bull_power = high_1d - ema13
    bear_power = low_1d - ema13
    
    # Align HTF Elder Ray to 6t timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # 6h EMA for entry timing
    close = prices['close'].values
    ema6 = pd.Series(close).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(ema6[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Bull Power positive and price above EMA6
            if bull_power_aligned[i] > 0 and price > ema6[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative and price below EMA6
            elif bear_power_aligned[i] < 0 and price < ema6[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bear Power negative (momentum shift) or price below EMA6
            if bear_power_aligned[i] < 0 or price < ema6[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bull Power positive (momentum shift) or price above EMA6
            if bull_power_aligned[i] > 0 or price > ema6[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_EMATrend_v1"
timeframe = "6h"
leverage = 1.0