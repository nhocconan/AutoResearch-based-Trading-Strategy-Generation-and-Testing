#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1-day Elder Ray (Bull/Bear Power) + 1-week trend filter
# Strategy: Long when Bull Power > 0 and weekly EMA50 > EMA200 (bullish regime)
# Short when Bear Power < 0 and weekly EMA50 < EMA200 (bearish regime)
# Elder Ray = Close - EMA13 (Bull Power) and EMA13 - Close (Bear Power)
# Uses 1-day EMA13 for power calculation and 1-week EMAs for regime filter
# Aims to capture momentum in trending markets while avoiding counter-trend whipsaws
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA13 on 1d for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power_1d = close_1d - ema13_1d  # Close - EMA13
    bear_power_1d = ema13_1d - close_1d   # EMA13 - Close
    
    # Get 1w data for regime filter (EMA50 and EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(ema200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: weekly EMA50 > EMA200 = bullish, < = bearish
        bullish_regime = ema50_1w_aligned[i] > ema200_1w_aligned[i]
        bearish_regime = ema50_1w_aligned[i] < ema200_1w_aligned[i]
        
        # Entry conditions
        long_entry = bull_power_aligned[i] > 0 and bullish_regime
        short_entry = bear_power_aligned[i] > 0 and bearish_regime  # Bear Power > 0 means bearish pressure
        
        # Exit conditions: opposite Elder Ray signal or regime change
        exit_long = position == 1 and (bear_power_aligned[i] > 0 or not bullish_regime)
        exit_short = position == -1 and (bull_power_aligned[i] > 0 or not bearish_regime)
        
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

name = "6h_1d_1w_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0