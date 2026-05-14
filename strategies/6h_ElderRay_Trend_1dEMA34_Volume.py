#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d EMA34 Trend + Volume Spike
# Elder Ray measures bull/bear power relative to EMA13: BullPower = High - EMA13, BearPower = Low - EMA13
# Long when BullPower > 0 and rising (momentum) + price > 1d EMA34 (uptrend) + volume spike
# Short when BearPower < 0 and falling + price < 1d EMA34 (downtrend) + volume spike
# Works in bull/bear by requiring trend alignment; volume reduces false signals.
# Target: 20-50 trades/year to minimize fee drag.

name = "6h_ElderRay_Trend_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h EMA13 for Elder Ray
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13_6h  # Higher = stronger bulls
    bear_power = low - ema_13_6h   # Lower = stronger bears (more negative)
    
    # Slope of Elder Ray (momentum) - using 3-period change
    bull_power_slope = bull_power - np.roll(bull_power, 3)
    bear_power_slope = bear_power - np.roll(bear_power, 3)
    # Handle roll boundary
    bull_power_slope[:3] = 0
    bear_power_slope[:3] = 0
    
    # 1d trend: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(bull_power_slope[i]) or 
            np.isnan(bear_power_slope[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: BullPower > 0 AND rising + uptrend (price > 1d EMA34) + volume spike
            long_cond = (bull_power[i] > 0) and \
                        (bull_power_slope[i] > 0) and \
                        (close[i] > ema_34_1d_aligned[i]) and \
                        volume_spike[i]
            # Short: BearPower < 0 AND falling + downtrend (price < 1d EMA34) + volume spike
            short_cond = (bear_power[i] < 0) and \
                         (bear_power_slope[i] < 0) and \
                         (close[i] < ema_34_1d_aligned[i]) and \
                         volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: BullPower turns negative (momentum loss)
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: BearPower turns positive (momentum loss)
            if bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals