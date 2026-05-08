#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with weekly trend filter and volume confirmation
# Long when price breaks above 20-period Donchian high (12h) + price > weekly EMA34 + volume > 1.5x 12h volume EMA20
# Short when price breaks below 20-period Donchian low (12h) + price < weekly EMA34 + volume > 1.5x 12h volume EMA20
# Donchian provides trend-following structure, weekly EMA34 filters long-term trend, volume confirms breakout strength
# Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year)

name = "12h_Donchian_Breakout_WeeklyEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 12h volume EMA20 for volume filter
    vol_ema_20_12h = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 12h Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align weekly EMA34 to 12h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian channels
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or np.isnan(vol_ema_20_12h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 12h volume > 1.5x 20-period EMA
        vol_filter = volume[i] > 1.5 * vol_ema_20_12h[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_max_20[i]
        breakout_down = close[i] < low_min_20[i]
        
        if position == 0:
            # Look for entry: Donchian breakout + weekly trend + volume
            long_condition = breakout_up and close[i] > ema_34_1w_aligned[i] and vol_filter
            short_condition = breakout_down and close[i] < ema_34_1w_aligned[i] and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low (20-period)
            if close[i] < low_min_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high (20-period)
            if close[i] > high_max_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals