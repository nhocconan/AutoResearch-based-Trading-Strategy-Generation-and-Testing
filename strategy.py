#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrendFilter_VolumeSpike
Hypothesis: Trade 6h Elder Ray Bull/Bear Power crossovers with 1d EMA50 trend filter and volume confirmation (>1.5x 20-bar MA). 
Elder Ray measures bull/bear power via EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13. 
Long when Bull Power crosses above zero AND Bear Power rising (less negative) AND 1d trend bullish AND volume spike. 
Short when Bear Power crosses below zero AND Bull Power falling (less positive) AND 1d trend bearish AND volume spike. 
Discrete sizing 0.25 balances profit and fee drag. Target: 12-25 trades/year (~50-100 over 4 years) to stay within fee drag limits.
Works in both bull and bear markets via 1d trend filter aligning with higher timeframe direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Elder Ray components: EMA13 of close, then Bull/Bear Power
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1d EMA50 (50) and EMA13 (13)
    start_idx = max(50, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power crosses above zero AND Bear Power rising (less negative) AND 1d trend bullish AND volume confirm
            bull_cross_up = (bull_power[i] > 0) and (bull_power[i-1] <= 0)
            bear_rising = bear_power[i] > bear_power[i-1]  # Bear Power becoming less negative
            long_setup = bull_cross_up and bear_rising and (close[i] > ema_50_1d_aligned[i]) and volume_confirm[i]
            
            # Short: Bear Power crosses below zero AND Bull Power falling (less positive) AND 1d trend bearish AND volume confirm
            bear_cross_down = (bear_power[i] < 0) and (bear_power[i-1] >= 0)
            bull_falling = bull_power[i] < bull_power[i-1]  # Bull Power becoming less positive
            short_setup = bear_cross_down and bull_falling and (close[i] < ema_50_1d_aligned[i]) and volume_confirm[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: Bull Power falls below zero OR 1d trend turns bearish
            if (bull_power[i] < 0) or (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: Bear Power rises above zero OR 1d trend turns bullish
            if (bear_power[i] > 0) or (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrendFilter_VolumeSpike"
timeframe = "6h"
leverage = 1.0