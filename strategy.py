#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA50 trend filter and volume spike confirmation
# Elder Ray measures bull/bear power via EMA13. Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Long when Bull Power > 0 and rising, Bear Power < 0 and falling, with 12h EMA50 uptrend and volume spike.
# Short when Bear Power < 0 and falling, Bull Power > 0 and rising, with 12h EMA50 downtrend and volume spike.
# Works in bull/bear: trend filter prevents counter-trend entries. Volume spike confirms institutional participation.
# Target: 50-150 total trades over 4 years = 12-37/year. HARD MAX: 300 total.

name = "6h_ElderRay_12hEMA50_VolumeSpike"
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
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend direction
    close_12h = pd.Series(df_12h['close'])
    ema50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 6h timeframe (completed 12h bar only)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate EMA13 for Elder Ray on 6h timeframe
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = ema13 - low   # Bear Power = EMA13 - Low
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirm = volume[i] > (2.0 * vol_ema_20[i])
        
        # Elder Ray momentum: Bull Power rising if current > previous, Bear Power falling if current < previous
        bull_power_rising = i > 0 and bull_power[i] > bull_power[i-1]
        bear_power_falling = i > 0 and bear_power[i] < bear_power[i-1]
        
        if position == 0:
            # Long conditions: Bull Power > 0 and rising, Bear Power < 0, uptrend, volume spike
            if (bull_power[i] > 0 and bull_power_rising and bear_power[i] < 0 and 
                close[i] > ema50_12h_aligned[i] and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 and falling, Bull Power > 0, downtrend, volume spike
            elif (bear_power[i] > 0 and bear_power_falling and bull_power[i] > 0 and 
                  close[i] < ema50_12h_aligned[i] and volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 OR Bear Power >= 0 OR trend changes OR volume drops
            if (bull_power[i] <= 0 or bear_power[i] >= 0 or 
                close[i] < ema50_12h_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power <= 0 OR Bull Power >= 0 OR trend changes OR volume drops
            if (bear_power[i] <= 0 or bull_power[i] >= 0 or 
                close[i] > ema50_12h_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals