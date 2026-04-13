#!/usr/bin/env python3
"""
6h_12h_EMA_Crossover_Trend_With_Volume_Filter
Hypothesis: Uses 12h EMA crossover (50/200) on 6h timeframe with volume confirmation to reduce whipsaws.
Enters long when 12h EMA50 crosses above EMA200 and 6h volume > 1.5x 20-period average.
Enters short when 12h EMA50 crosses below EMA200 and 6h volume > 1.5x 20-period average.
Exits on opposite crossover.
Designed for 6h timeframe to target 12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets by requiring volume expansion on trend changes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 and EMA200
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 20-period volume average on 12h
    vol_ma_20_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all signals to 6h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Detect crossovers
    ema50_above_200 = ema50_12h_aligned > ema200_12h_aligned
    ema50_below_200 = ema50_12h_aligned < ema200_12h_aligned
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(ema200_12h_aligned[i]) or 
            np.isnan(vol_ma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 12h volume MA
        volume_expansion = volume[i] > (vol_ma_20_12h_aligned[i] * 1.5)
        
        # Entry conditions: EMA crossover with volume expansion
        long_entry = ema50_above_200[i] and not ema50_above_200[i-1] and volume_expansion
        short_entry = ema50_below_200[i] and not ema50_below_200[i-1] and volume_expansion
        
        # Exit conditions: opposite crossover
        exit_long = position == 1 and ema50_below_200[i] and not ema50_below_200[i-1]
        exit_short = position == -1 and ema50_above_200[i] and not ema50_above_200[i-1]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_EMA_Crossover_Trend_With_Volume_Filter"
timeframe = "6h"
leverage = 1.0