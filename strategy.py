#!/usr/bin/env python3
"""
4h_1d_1w_ElderRay_Momentum_Breakout_V1
Hypothesis: Elder Ray bull/bear power breaks above/below zero with 1w trend confirmation and volume filter.
Long when bull power crosses above zero with 1w close > 200 EMA and volume spike.
Short when bear power crosses below zero with 1w close < 200 EMA and volume spike.
Exit when power crosses back through zero or price reaches 1w ATR-based stop.
Works in both bull/bear by following 1w trend and using momentum exhaustion signals.
Target: 20-35 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Elder Ray (13-period EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA for Elder Ray
    close_ser = pd.Series(close_1d)
    ema13 = close_ser.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema13
    bear_power = low_1d - ema13
    
    # Align to 4h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Load 1w data for trend filter (200 EMA)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    close_1w_ser = pd.Series(close_1w)
    ema200_1w = close_1w_ser.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1w EMA200 to 4h timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 2.0 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: bull power crosses above zero with 1w uptrend and volume
            if (bull_power_aligned[i] > 0 and bull_power_aligned[i-1] <= 0 and  # crossover up
                price > ema200_1w_aligned[i] and volume_ok):
                signals[i] = 0.25
                position = 1
            # Short conditions: bear power crosses below zero with 1w downtrend and volume
            elif (bear_power_aligned[i] < 0 and bear_power_aligned[i-1] >= 0 and  # crossover down
                  price < ema200_1w_aligned[i] and volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bull power crosses below zero or trend fails
            if bull_power_aligned[i] < 0 or price < ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bear power crosses above zero or trend fails
            if bear_power_aligned[i] > 0 or price > ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_1w_ElderRay_Momentum_Breakout_V1"
timeframe = "4h"
leverage = 1.0