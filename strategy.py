#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly EMA200 trend filter with daily Donchian(20) breakout and volume confirmation
# Long when price breaks above Donchian high + price > weekly EMA200 + volume > 1.5x 20-day average volume
# Short when price breaks below Donchian low + price < weekly EMA200 + volume > 1.5x 20-day average volume
# Exit when price crosses weekly EMA200 in opposite direction
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag

name = "1d_Donchian_WeeklyEMA200_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter
    ema_200 = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly EMA200 to daily timeframe
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    # Calculate daily Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-day average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(ema_200_aligned[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current volume > 1.5x 20-day average volume
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Look for entry: Donchian breakout + trend + volume
            long_condition = close[i] > high_max[i] and close[i] > ema_200_aligned[i] and vol_filter
            short_condition = close[i] < low_min[i] and close[i] < ema_200_aligned[i] and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly EMA200
            if close[i] < ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above weekly EMA200
            if close[i] > ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals