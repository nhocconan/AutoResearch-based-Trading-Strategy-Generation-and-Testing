#!/usr/bin/env python3
"""
6h_Pivot_R1S1_Fade_v1
1d Camarilla pivot fade strategy on 6h timeframe.
Fade at R1/S1 with momentum confirmation, breakout continuation at R4/S4.
Uses 1d pivot levels for structure, 60-period EMA for momentum filter.
Targets 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 60-period EMA for momentum filter ===
    alpha = 2 / (60 + 1)
    ema60 = np.zeros_like(close)
    ema60[0] = close[0]
    for i in range(1, n):
        ema60[i] = ema60[i-1] + alpha * (close[i] - ema60[i-1])
    
    # === Daily Camarilla pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    # Typical price for pivot calculation
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    pivot = typical_price.values
    # Calculate ranges
    range_high_low = df_1d['high'].values - df_1d['low'].values
    # Camarilla levels
    r1 = pivot + 0.1166 * range_high_low
    s1 = pivot - 0.1166 * range_high_low
    r4 = pivot + 0.5500 * range_high_low
    s4 = pivot - 0.5500 * range_high_low
    
    # Align pivot levels to 6h timeframe (with 1-bar delay for completed daily bar)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema60[i]) or 
            np.isnan(r1_6h[i]) or 
            np.isnan(s1_6h[i]) or 
            np.isnan(r4_6h[i]) or 
            np.isnan(s4_6h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Fade at R1/S1: price rejects pivot level with momentum
            # Long: price crosses below S1 but closes above it with bullish momentum
            if (low[i] <= s1_6h[i] and 
                close[i] > s1_6h[i] and 
                close[i] > ema60[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price crosses above R1 but closes below it with bearish momentum
            elif (high[i] >= r1_6h[i] and 
                  close[i] < r1_6h[i] and 
                  close[i] < ema60[i]):
                signals[i] = -0.25
                position = -1
                continue
            # Breakout continuation at R4/S4: strong momentum breaks key levels
            # Long: breaks above R4 with momentum
            elif (high[i] >= r4_6h[i] and 
                  close[i] > r4_6h[i] and 
                  close[i] > ema60[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: breaks below S4 with momentum
            elif (low[i] <= s4_6h[i] and 
                  close[i] < s4_6h[i] and 
                  close[i] < ema60[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price returns to pivot or momentum fails
            if (close[i] <= pivot[i] or 
                close[i] < ema60[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot or momentum fails
            if (close[i] >= pivot[i] or 
                close[i] > ema60[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_R1S1_Fade_v1"
timeframe = "6h"
leverage = 1.0