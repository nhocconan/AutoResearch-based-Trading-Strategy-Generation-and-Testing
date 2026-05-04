#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h trend filter and 1d volume spike confirmation
# In 4h uptrend (price > 20 EMA): long when 1h price breaks above 20-period Donchian high + 1d volume > 1.5x 20 EMA
# In 4h downtrend (price < 20 EMA): short when 1h price breaks below 20-period Donchian low + 1d volume > 1.5x 20 EMA
# Uses discrete sizing (0.20) to minimize fees and session filter (08-20 UTC) to reduce noise.
# Designed for 1h timeframe targeting 60-150 total trades over 4 years (15-37/year).
# BTC/ETH edge: Donchian breakouts capture momentum; 4h EMA filter avoids counter-trend trades; volume confirms institutional participation.

name = "1h_Donchian20_4hEMA20_1dVolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 arithmetic errors)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h 20-period EMA
    ema_20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d 20-period EMA of volume
    vol_ema_20_1d = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20_1d)
    
    # Calculate 1h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(vol_ema_20_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session and volume confirmation
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        volume_confirm = volume[i] > (1.5 * vol_ema_20_1d_aligned[i])
        
        if position == 0:
            # Determine 4h trend: bullish (price > EMA20) or bearish (price < EMA20)
            if close[i] > ema_20_4h_aligned[i]:
                # Bullish trend: look for long breakout
                if high[i] > highest_high[i] and volume_confirm:
                    signals[i] = 0.20
                    position = 1
            else:
                # Bearish trend: look for short breakout
                if low[i] < lowest_low[i] and volume_confirm:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Exit long: price closes below 4h EMA20 OR Donchian mid-channel
            if close[i] < ema_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above 4h EMA20 OR Donchian mid-channel
            if close[i] > ema_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals