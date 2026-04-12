#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_12h_elder_ray_regime_v1
# Uses 12h Elder Ray (Bull/Bear Power) with 6h EMA(50) trend filter.
# Long when Bull Power > 0 and price above 6h EMA(50).
# Short when Bear Power < 0 and price below 6h EMA(50).
# Exits when Elder Power crosses zero or price crosses EMA(50).
# Designed for low trade frequency (target: 12-37 trades/year) to minimize fee drag.
# Works in trending markets via trend alignment and momentum exhaustion via Elder Ray zero-cross.

name = "6h_12h_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for Elder Ray calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(13) for Elder Ray
    close_12h_series = pd.Series(close_12h)
    ema13_12h = close_12h_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_12h - ema13_12h
    bear_power = low_12h - ema13_12h
    
    # Align 12h Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)
    
    # 6h EMA(50) for trend filter
    close_series = pd.Series(close)
    ema50_6h = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(ema50_6h[i]):
            signals[i] = 0.0
            continue
        
        # Long: Bull Power > 0 and price above EMA50
        if bull_power_aligned[i] > 0 and close[i] > ema50_6h[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Bear Power < 0 and price below EMA50
        elif bear_power_aligned[i] < 0 and close[i] < ema50_6h[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Elder Power crosses zero or price crosses EMA50
        elif position == 1 and (bull_power_aligned[i] <= 0 or close[i] <= ema50_6h[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bear_power_aligned[i] >= 0 or close[i] >= ema50_6h[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals