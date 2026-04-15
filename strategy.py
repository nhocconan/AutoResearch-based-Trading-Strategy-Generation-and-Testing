#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h trend filter and 1d range breakout for entry timing
# Uses 4h EMA(20) for trend direction, 1d high/low breakout for entry, and volume confirmation
# Designed to work in both bull and bear markets by following higher timeframe trend
# Target: 60-150 total trades over 4 years (15-37/year) with session filter (08-20 UTC)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data for trend filter (EMA20)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate EMA(20) on 4h
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 4h EMA to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Load 1d data for range breakout levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's high and low (avoid look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    # Align 1d levels to 1h timeframe
    prev_high_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    
    # Volume filter: 1.5x 20-period median
    volume_median = pd.Series(volume).rolling(window=20, min_periods=1).median().values
    
    # Session filter: 08-20 UTC (precompute hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.20  # Position size
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_aligned[i]) or 
            np.isnan(prev_high_1d_aligned[i]) or 
            np.isnan(prev_low_1d_aligned[i])):
            continue
            
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            continue
        
        # Long entry: price above 4h EMA (uptrend) + breaks above prev day high + volume
        if (close[i] > ema_4h_aligned[i] and
            close[i] > prev_high_1d_aligned[i] and
            volume[i] > 1.5 * volume_median[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price below 4h EMA (downtrend) + breaks below prev day low + volume
        elif (close[i] < ema_4h_aligned[i] and
              close[i] < prev_low_1d_aligned[i] and
              volume[i] > 1.5 * volume_median[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse trend or break opposite level
        elif position == 1 and (close[i] < ema_4h_aligned[i] or close[i] < prev_low_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > ema_4h_aligned[i] or close[i] > prev_high_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1h_4hEMA_1dBreakout_Volume_Session"
timeframe = "1h"
leverage = 1.0