#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 12h EMA34 trend filter and volume spike confirmation
# Elder Ray measures bull/bear power relative to EMA13. In strong trends (price > 12h EMA34),
# we take reversals from extreme Elder Ray levels for trend continuation. Volume spike (>2x 20 EMA) confirms.
# Discrete sizing 0.25 limits risk. Target: 50-150 trades over 4 years (12-37/year).
# Works in bull/bear: uses 12h trend filter to align with higher timeframe direction.

name = "6h_ElderRay_12hEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

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
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend direction
    close_12h = pd.Series(df_12h['close'])
    ema34_12h = close_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA34 to 6h timeframe (completed 12h bar only)
    ema34_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate Elder Ray Index on 6h timeframe
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirm = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long conditions: extreme bear power (selling exhaustion) + uptrend + volume spike
            if bear_power[i] <= np.percentile(bear_power[max(0, i-100):i+1], 5) and close[i] > ema34_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: extreme bull power (buying exhaustion) + downtrend + volume spike
            elif bull_power[i] >= np.percentile(bull_power[max(0, i-100):i+1], 95) and close[i] < ema34_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bear power normalizes OR trend changes OR volume drops
            if (bear_power[i] >= np.percentile(bear_power[max(0, i-100):i+1], 50) or 
                close[i] < ema34_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bull power normalizes OR trend changes OR volume drops
            if (bull_power[i] <= np.percentile(bull_power[max(0, i-100):i+1], 50) or 
                close[i] > ema34_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals