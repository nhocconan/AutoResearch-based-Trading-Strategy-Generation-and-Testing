#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_12hTrendFilter_VolumeSpike
Hypothesis: Trade 6h timeframe using Elder Ray Bull/Bear Power filtered by 12h EMA trend and volume spikes.
Enter long when Bear Power crosses above zero (bullish momentum building) AND 12h trend bullish (close > EMA50) AND volume > 1.5x 20-period average.
Enter short when Bull Power crosses below zero (bearish momentum building) AND 12h trend bearish (close < EMA50) AND volume spike.
Exit when Elder Ray power reverses or 12h trend changes.
Uses discrete sizing 0.25 to manage risk and minimize fee churn. Target 15-35 trades/year on 6h timeframe.
Elder Ray measures bull/bear power relative to EMA13, capturing momentum shifts before price breaks.
12h EMA50 filter ensures we only trade with the higher timeframe trend, reducing counter-trend whipsaws.
Volume spike confirmation adds conviction to momentum shifts, filtering out false signals.
Works in both bull and bear markets by trading with the 12h trend while capturing momentum exhaustion/reversal signals.
"""

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
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Elder Ray Bull/Bear Power (requires EMA13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 12h EMA50 (50) and EMA13 (13) and volume MA (20)
    start_idx = max(50, 13, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bear Power crosses above zero (from negative to positive) AND 12h trend bullish AND volume spike
            bullish_momentum = (bear_power[i] > 0) and (bear_power[i-1] <= 0)
            long_setup = bullish_momentum and \
                         (close[i] > ema_50_12h_aligned[i]) and \
                         volume_spike[i]
            # Short: Bull Power crosses below zero (from positive to negative) AND 12h trend bearish AND volume spike
            bearish_momentum = (bull_power[i] < 0) and (bull_power[i-1] >= 0)
            short_setup = bearish_momentum and \
                          (close[i] < ema_50_12h_aligned[i]) and \
                          volume_spike[i]
            
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
            # Exit: Bear Power crosses back below zero (momentum fading) OR 12h trend turns bearish
            if (bear_power[i] < 0 and bear_power[i-1] >= 0) or \
               (close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: Bull Power crosses back above zero (momentum fading) OR 12h trend turns bullish
            if (bull_power[i] > 0 and bull_power[i-1] <= 0) or \
               (close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_12hTrendFilter_VolumeSpike"
timeframe = "6h"
leverage = 1.0