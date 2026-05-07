#!/usr/bin/env python3
name = "6h_ElderRay_BullBearPower_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray (EMA13) and trend filter (EMA100)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d EMA100 for trend filter
    ema100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = high - ema13_1d
    bear_power = low - ema13_1d
    
    # Align indicators to 6h timeframe (wait for completed 1d candle)
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema13_1d_aligned[i]) or 
            np.isnan(ema100_1d_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 and price above EMA100 (uptrend)
            if bull_power_aligned[i] > 0 and close[i] > ema100_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 and price below EMA100 (downtrend)
            elif bear_power_aligned[i] < 0 and close[i] < ema100_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power < 0 (momentum shift) or price below EMA100 (trend change)
            if bear_power_aligned[i] < 0 or close[i] < ema100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power > 0 (momentum shift) or price above EMA100 (trend change)
            if bull_power_aligned[i] > 0 or close[i] > ema100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals