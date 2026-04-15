#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 12h EMA50 Trend Filter + Volume Spike
# Elder Ray measures bull/bear power relative to EMA13. 
# Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Trades in direction of 12h EMA50 trend when power is strong and volume spikes.
# Works in bull markets (strong bull power) and bear markets (strong bear power).
# Target: 50-150 total trades over 4 years = 12-37/year.
# Timeframe: 6h, HTF: 12h for trend filter.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray (6h)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power and Bear Power
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h
    close_12h_s = pd.Series(close_12h)
    ema50_12h = close_12h_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50_12h to 6h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(13, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_12h_aligned[i])):
            continue
        
        # Volume spike: current volume > 2.0 * median of last 20 periods
        vol_median = np.median(volume[max(0, i-20):i+1])
        volume_spike = volume[i] > 2.0 * vol_median
        
        # Long entry: bull power positive + price above 12h EMA50 + volume spike
        if (bull_power[i] > 0 and 
            close[i] > ema50_12h_aligned[i] and 
            volume_spike and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: bear power positive + price below 12h EMA50 + volume spike
        elif (bear_power[i] > 0 and 
              close[i] < ema50_12h_aligned[i] and 
              volume_spike and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: power diminishes or opposite power becomes strong
        elif position == 1 and (bull_power[i] <= 0 or bear_power[i] > bull_power[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bear_power[i] <= 0 or bull_power[i] > bear_power[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_12hEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0