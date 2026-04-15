#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4h-1d Trend Filter with Volume Spike and Session Filter
# Uses 4h EMA for trend direction, 1d Donchian breakout for entry timing, volume spike for confirmation.
# Designed for 1h timeframe: trend from 4h, entry from 1d breakout, volume filter to avoid false signals.
# Session filter (08-20 UTC) reduces noise. Target: 15-37 trades/year (60-150 over 4 years).
# Works in bull markets (trend up + breakout up) and bear markets (trend down + breakout down).

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Load 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 4h EMA(50) for trend
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d Donchian channels (20-period)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align indicators to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Volume spike filter: volume > 2.0 * 20-period median volume
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.20  # Position size
    
    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or np.isnan(vol_median[i])):
            continue
        
        # Check session: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            continue
        
        # Long entry: 4h EMA up (trend) + price breaks above 1d Donchian high + volume spike
        if (close[i] > ema_4h_aligned[i] and  # Uptrend filter
            close[i] > high_20_aligned[i] and  # Breakout above 1d high
            volume[i] > 2.0 * vol_median[i] and  # Volume spike
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: 4h EMA down (trend) + price breaks below 1d Donchian low + volume spike
        elif (close[i] < ema_4h_aligned[i] and  # Downtrend filter
              close[i] < low_20_aligned[i] and  # Breakout below 1d low
              volume[i] > 2.0 * vol_median[i] and  # Volume spike
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: trend reversal or volatility contraction
        elif position == 1 and close[i] < ema_4h_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > ema_4h_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1h_4hEMA_1dDonchian_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0