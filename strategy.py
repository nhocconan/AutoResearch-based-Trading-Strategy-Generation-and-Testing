#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_Trend_v1
Elder Ray Index: Bull Power = High - EMA13, Bear Power = EMA13 - Low.
Long when Bull Power > 0 and rising, Bear Power < 0 and falling.
Short when Bear Power < 0 and falling, Bull Power > 0 and rising.
Uses 1d trend filter: price above/below 1d EMA50.
Exit when power crosses zero or trend fails.
Designed to capture institutional buying/selling pressure with trend alignment.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === EMA13 for Elder Ray ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # === Elder Ray Components ===
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 20
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Bull Power > 0 and rising, Bear Power < 0, price above 1d EMA50
            if (bull_power[i] > 0 and 
                bull_power[i] > bull_power[i-1] and 
                bear_power[i] < 0 and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Bear Power < 0 and falling, Bull Power > 0, price below 1d EMA50
            elif (bear_power[i] < 0 and 
                  bear_power[i] < bear_power[i-1] and 
                  bull_power[i] > 0 and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Bull Power <= 0 OR Bear Power >= 0 OR price below 1d EMA50
            if (bull_power[i] <= 0 or 
                bear_power[i] >= 0 or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power >= 0 OR Bull Power <= 0 OR price above 1d EMA50
            if (bear_power[i] >= 0 or 
                bull_power[i] <= 0 or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_Trend_v1"
timeframe = "6h"
leverage = 1.0