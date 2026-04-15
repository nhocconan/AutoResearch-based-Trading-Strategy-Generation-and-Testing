#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power with 12h EMA50 Trend Filter and Volume Confirmation
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 and increasing + price > 12h EMA50 + volume > 1.5x median
# Short when Bear Power < 0 and decreasing + price < 12h EMA50 + volume > 1.5x median
# Exit when power crosses zero or volume drops
# Works in bull markets (strong bull power) and bear markets (strong bear power)
# Target: 50-150 total trades over 4 years = 12-37/year. Well under 300 max.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate EMA13 on 6h for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray Power
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate EMA50 on 12h for trend filter
    close_12h_s = pd.Series(close_12h)
    ema50_12h = close_12h_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50_12h to 6h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25%)
    
    for i in range(13, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_12h_aligned[i])):
            continue
        
        # Calculate 20-period median volume for confirmation
        if i >= 20:
            vol_median = np.median(volume[i-20:i])
        else:
            vol_median = np.median(volume[:i+1]) if i > 0 else volume[i]
        
        vol_confirm = volume[i] > 1.5 * vol_median
        
        # Long entry: Bull Power positive AND increasing + price > 12h EMA50 + volume confirmation
        if (bull_power[i] > 0 and 
            bull_power[i] > bull_power[i-1] and  # Increasing bull power
            close[i] > ema50_12h_aligned[i] and
            vol_confirm and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Bear Power negative AND decreasing + price < 12h EMA50 + volume confirmation
        elif (bear_power[i] < 0 and 
              bear_power[i] < bear_power[i-1] and  # Decreasing bear power (more negative)
              close[i] < ema50_12h_aligned[i] and
              vol_confirm and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Power crosses zero or volume drops significantly
        elif position == 1 and (bull_power[i] <= 0 or volume[i] < 0.5 * vol_median):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bear_power[i] >= 0 or volume[i] < 0.5 * vol_median):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0