#!/usr/bin/env python3
"""
6h_ElderRay_Slope_ZeroCross_V2
Hypothesis: Elder Ray (Bull/Bear Power) with EMA13 slope and zero-cross exit.
Long when Bull Power > 0 and EMA13 slope turns up. Short when Bear Power < 0 and EMA13 slope turns down.
Exit when Bull/Bear Power crosses zero or EMA13 slope reverses.
Uses 1d EMA200 for trend filter to avoid counter-trend trades.
Designed for 6h timeframe with ~15-35 trades/year.
"""

name = "6h_ElderRay_Slope_ZeroCross_V2"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate EMA13 for slope (on 6h chart)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate EMA13 slope (change over 3 periods)
    ema13_slope = np.zeros_like(ema_13)
    ema13_slope[3:] = ema_13[3:] - ema_13[:-3]
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 13)  # Warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_200_1d_aligned[i]) or np.isnan(ema13_slope[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA200
        price_above_ema200 = close[i] > ema_200_1d_aligned[i]
        price_below_ema200 = close[i] < ema_200_1d_aligned[i]
        
        # Slope direction
        slope_up = ema13_slope[i] > 0
        slope_down = ema13_slope[i] < 0
        
        if position == 0:
            # Long: Bull Power > 0, EMA13 slope up, price above 1d EMA200
            if (bull_power[i] > 0 and 
                slope_up and 
                price_above_ema200):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, EMA13 slope down, price below 1d EMA200
            elif (bear_power[i] < 0 and 
                  slope_down and 
                  price_below_ema200):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power crosses zero OR slope turns down
            if (bull_power[i] <= 0 or 
                slope_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power crosses zero OR slope turns up
            if (bear_power[i] >= 0 or 
                slope_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals