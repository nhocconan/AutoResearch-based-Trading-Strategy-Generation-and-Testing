#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation
# Elder Ray measures bull/bear power via EMA13 deviation: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 and rising, Bear Power < 0 and falling, price > 1d EMA34 (uptrend), volume > 1.5x 20 EMA
# Short when Bear Power < 0 and falling, Bull Power > 0 and rising, price < 1d EMA34 (downtrend), volume confirmation
# Uses discrete sizing 0.25 to limit drawdown. Works in bull/bear via trend filter. Target: 50-150 trades over 4 years.

name = "6h_ElderRay_1dEMA34_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend direction
    close_1d = pd.Series(df_1d['close'])
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe (completed 1d bar only)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate EMA13 for Elder Ray on 6h timeframe
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Calculate rate of change for Elder Ray components (to detect rising/falling)
    bull_power_change = np.diff(bull_power, prepend=bull_power[0])
    bear_power_change = np.diff(bear_power, prepend=bear_power[0])
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_aligned[i]) or np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long conditions: Bull Power > 0 and rising, Bear Power < 0, uptrend, volume spike
            if (bull_power[i] > 0 and bull_power_change[i] > 0 and 
                bear_power[i] < 0 and close[i] > ema34_aligned[i] and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 and falling, Bull Power > 0, downtrend, volume spike
            elif (bear_power[i] < 0 and bear_power_change[i] < 0 and 
                  bull_power[i] > 0 and close[i] < ema34_aligned[i] and volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power turns negative OR Bear Power rises above zero OR trend changes OR volume drops
            if (bull_power[i] <= 0 or bear_power[i] >= 0 or 
                close[i] < ema34_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power turns positive OR Bull Power falls below zero OR trend changes OR volume drops
            if (bear_power[i] >= 0 or bull_power[i] <= 0 or 
                close[i] > ema34_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals