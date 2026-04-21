#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_v1
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) with 6h zero-cross signals.
Long when Bull Power crosses above zero with rising Bear Power (less negative) = bullish momentum.
Short when Bear Power crosses above zero with falling Bull Power (less positive) = bearish momentum.
Uses 1d EMA50 trend filter and volume confirmation. Works in bull/bear: adapts to momentum shifts.
Target: 12-25 trades/year per symbol (50-100 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 6h data for Elder Ray calculation
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA13 for Elder Ray (6h timeframe)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema_13
    # Bear Power = EMA13 - Low
    bear_power = ema_13 - low
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if indicators not ready
        if np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current volume > 1.3 * 20-period average
        if i >= 20:
            vol_ma = volume[i-20:i].mean()
            volume_ok = volume[i] > 1.3 * vol_ma
        else:
            volume_ok = False
        
        # Determine 1d trend: EMA50 rising/falling
        if i > 0:
            ema_50_rising = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            ema_50_falling = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]
        else:
            ema_50_rising = True
            ema_50_falling = False
        
        if position == 0:
            # Long: Bull Power crosses above zero AND Bear Power rising (less negative) AND uptrend AND volume
            bull_cross_up = (bull_power[i-1] <= 0 and bull_power[i] > 0) if i > 0 else False
            bear_power_rising = (i > 0 and bear_power[i] > bear_power[i-1])
            
            if bull_cross_up and bear_power_rising and ema_50_rising and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power crosses above zero AND Bull Power falling (less positive) AND downtrend AND volume
            elif (i > 0 and bear_power[i-1] <= 0 and bear_power[i] > 0 and  # Bear Power cross up
                  bull_power[i] < bull_power[i-1] and  # Bull Power falling
                  ema_50_falling and volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power crosses below zero OR Bear Power falls sharply
            bull_cross_down = (bull_power[i-1] > 0 and bull_power[i] <= 0) if i > 0 else False
            bear_power_falling_sharp = (i > 0 and bear_power[i] < bear_power[i-1] * 0.8)  # 20% drop
            
            if bull_cross_down or bear_power_falling_sharp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power crosses below zero OR Bull Power rises sharply
            bear_cross_down = (bear_power[i-1] > 0 and bear_power[i] <= 0) if i > 0 else False
            bull_power_rising_sharp = (i > 0 and bull_power[i] > bull_power[i-1] * 1.2)  # 20% rise
            
            if bear_cross_down or bull_power_rising_sharp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_v1"
timeframe = "6h"
leverage = 1.0