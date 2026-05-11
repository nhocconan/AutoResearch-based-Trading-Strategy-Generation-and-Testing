#!/usr/bin/env python3
name = "6h_ElderRay_BullBearPower_1dTrend"
timeframe = "6h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate 13-period EMA on 1d close for trend
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components on 1d data
    # Bull Power = High - EMA13
    bull_power = high - ema13_1d
    # Bear Power = Low - EMA13  
    bear_power = low - ema13_1d
    
    # Align Elder Ray components and EMA13 to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Volume filter: 20-period average on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema13_aligned[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold - avoid low-volume false signals
        volume_surge = vol_ratio[i] > 1.3
        
        if position == 0:
            # Long: Bull Power > 0 (strength) AND price above EMA13 (uptrend) with volume
            if (bull_power_aligned[i] > 0 and 
                close[i] > ema13_aligned[i] and 
                volume_surge):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (weakness) AND price below EMA13 (downtrend) with volume
            elif (bear_power_aligned[i] < 0 and 
                  close[i] < ema13_aligned[i] and 
                  volume_surge):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: Bear Power turns negative (momentum loss) OR price crosses below EMA13
                if (bear_power_aligned[i] < 0) or (close[i] < ema13_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: Bull Power turns positive (momentum loss) OR price crosses above EMA13
                if (bull_power_aligned[i] > 0) or (close[i] > ema13_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals