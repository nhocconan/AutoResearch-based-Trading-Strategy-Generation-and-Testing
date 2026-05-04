#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian(20) breakout with 4h EMA(50) trend filter and volume spike (>2.0x 20 EMA volume)
# Uses 4h EMA for direction to avoid 1h whipsaw, Donchian for precise entry timing on 1h
# Volume confirmation ensures breakout has conviction
# Session filter (08-20 UTC) reduces noise trades
# Discrete sizing 0.20 balances risk and return while minimizing fee churn
# Target: 60-120 total trades over 4 years = 15-30/year for 1h timeframe
# Uses 4h/1d for signal direction, 1h only for entry timing as per instructions

name = "1h_Donchian20_4hEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need enough data for EMA calculation
        return np.zeros(n)
    
    # Calculate 4h EMA(50) trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate Donchian channels (20-period) on 1h timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period EMA of volume on 1h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ema_20[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper + 4h EMA uptrend + volume spike
            if close[i] > highest_high[i] and close[i] > ema_4h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Donchian lower + 4h EMA downtrend + volume spike
            elif close[i] < lowest_low[i] and close[i] < ema_4h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR 4h EMA trend reverses
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < midpoint or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR 4h EMA trend reverses
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > midpoint or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals