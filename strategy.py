#!/usr/bin/env python3
"""
6h_1d_elder_ray_power_zone_v1
Strategy: 6h Elder Ray Power Zone with 1d EMA trend filter
Timeframe: 6h
Leverage: 1.0
Hypothesis: Uses 6h Elder Ray (bull/bear power) to detect momentum exhaustion and reversals. 
Bull Power = High - EMA(13), Bear Power = EMA(13) - Low. 
Enter long when Bull Power turns positive after being negative (momentum shift up) 
and price is above 1d EMA50 (uptrend filter). 
Enter short when Bear Power turns positive after being negative (momentum shift down) 
and price is below 1d EMA50 (downtrend filter).
Exit when power returns to zero or opposite signal occurs.
Designed to capture momentum shifts in both bull and bear markets by focusing on 
institutional buying/selling pressure rather than price alone. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_elder_ray_power_zone_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 6h EMA13 for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Elder Ray components
    bull_power = high - ema_13  # Buying power
    bear_power = ema_13 - low   # Selling power
    
    # Previous values for crossover detection
    bull_power_prev = np.roll(bull_power, 1)
    bear_power_prev = np.roll(bear_power, 1)
    bull_power_prev[0] = np.nan
    bear_power_prev[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_13[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(bull_power_prev[i]) or np.isnan(bear_power_prev[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter from 1d EMA50
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # Elder Ray signals: momentum shift detection
        # Bullish: Bull Power crosses above zero (buying pressure emerging)
        bull_crossover = bull_power_prev[i] <= 0 and bull_power[i] > 0
        # Bearish: Bear Power crosses above zero (selling pressure emerging) 
        bear_crossover = bear_power_prev[i] <= 0 and bear_power[i] > 0
        
        # Long: bullish momentum shift in uptrend
        long_signal = bull_crossover and uptrend_1d
        
        # Short: bearish momentum shift in downtrend
        short_signal = bear_crossover and downtrend_1d
        
        # Exit conditions
        exit_long = position == 1 and (bull_power[i] <= 0 or bear_crossover)  # Long exit: buying pressure fades or selling pressure emerges
        exit_short = position == -1 and (bear_power[i] <= 0 or bull_crossover)  # Short exit: selling pressure fades or buying pressure emerges
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals