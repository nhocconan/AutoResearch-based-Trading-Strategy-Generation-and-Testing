#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power + 12h EMA50 Trend Filter + Volume Spike
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 and Bear Power < 0 and price > 12h EMA50 and volume > 1.5x median
# Short when Bear Power < 0 and Bull Power < 0 and price < 12h EMA50 and volume > 1.5x median
# Uses 12h EMA50 for trend filter to avoid counter-trend trades
# Volume spike confirms institutional interest
# Target: 50-150 total trades over 4 years = 12-37/year
# Timeframe: 6h, HTF: 12h

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power and Bear Power
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h
    close_12h_s = pd.Series(close_12h)
    ema50_12h = close_12h_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 6h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema50_12h_aligned[i])):
            continue
        
        # Volume condition: current volume > 1.5x median of last 20 bars
        vol_median = np.median(volume[max(0, i-20):i+1])
        volume_ok = volume[i] > 1.5 * vol_median
        
        # Long entry: Bull Power > 0, Bear Power < 0, price > EMA50, volume spike
        if (bull_power[i] > 0 and bear_power[i] < 0 and
            close[i] > ema50_12h_aligned[i] and volume_ok and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Bear Power < 0, Bull Power < 0, price < EMA50, volume spike
        elif (bear_power[i] < 0 and bull_power[i] < 0 and
              close[i] < ema50_12h_aligned[i] and volume_ok and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Power signals reverse or trend fails
        elif position == 1 and (bull_power[i] <= 0 or bear_power[i] >= 0 or
                                close[i] <= ema50_12h_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bull_power[i] >= 0 or bear_power[i] >= 0 or
                                 close[i] >= ema50_12h_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0