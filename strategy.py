#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d volume filter + 1w EMA trend filter
# Elder Ray measures bull/bear power using EMA13 as reference: 
# Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 and rising + volume above average + price above weekly EMA50
# Short when Bear Power < 0 and falling + volume above average + price below weekly EMA50
# Works in bull/bear by using weekly EMA for trend and Elder Ray for momentum/strength.
# Target: 80-160 total trades over 4 years (20-40/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate EMA13 for Elder Ray (1d)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = high_1d - ema13_1d  # Higher = stronger bulls
    bear_power = low_1d - ema13_1d   # Lower (more negative) = stronger bears
    
    # Calculate EMA50 on weekly for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period on 1d)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power)
    ema50_1w_6h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    vol_avg_6h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or
            np.isnan(ema50_1w_6h[i]) or np.isnan(vol_avg_6h[i])):
            continue
        
        # Long entry: Bull Power > 0 and rising (bulls gaining strength) + volume spike + price above weekly EMA50
        if (bull_power_6h[i] > 0 and 
            bull_power_6h[i] > bull_power_6h[i-1] and  # Rising bull power
            volume[i] > 1.5 * vol_avg_6h[i] and
            close[i] > ema50_1w_6h[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Bear Power < 0 and falling (bears gaining strength) + volume spike + price below weekly EMA50
        elif (bear_power_6h[i] < 0 and 
              bear_power_6h[i] < bear_power_6h[i-1] and  # Falling bear power (more negative)
              volume[i] > 1.5 * vol_avg_6h[i] and
              close[i] < ema50_1w_6h[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or power crosses zero (momentum shift)
        elif position == 1 and (bull_power_6h[i] <= 0 or bear_power_6h[i] >= 0):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bear_power_6h[i] >= 0 or bull_power_6h[i] <= 0):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_Volume_WeeklyTrend"
timeframe = "6h"
leverage = 1.0