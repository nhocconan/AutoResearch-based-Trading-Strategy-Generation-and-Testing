#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA34 trend filter and volume spike confirmation
# Elder Ray measures bull/bear power relative to EMA13. In strong trends (price > 12h EMA34),
# we take long when bull power > 0 and volume spikes, short when bear power < 0 and volume spikes.
# Works in bull/bear: uses 12h EMA34 trend filter to align with higher timeframe direction.
# Discrete sizing 0.25 limits risk. Target: 50-150 trades over 4 years (12-37/year).

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
    
    # Calculate EMA13 for Elder Ray on 6h timeframe
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
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
            # Long conditions: bull power positive + uptrend + volume spike
            if bull_power[i] > 0 and close[i] > ema34_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: bear power negative + downtrend + volume spike
            elif bear_power[i] < 0 and close[i] < ema34_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bull power turns negative OR trend changes OR volume drops
            if (bull_power[i] <= 0 or 
                close[i] < ema34_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bear power turns positive OR trend changes OR volume drops
            if (bear_power[i] >= 0 or 
                close[i] > ema34_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals