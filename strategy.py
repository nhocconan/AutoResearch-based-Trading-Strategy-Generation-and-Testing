#!/usr/bin/env python3
# 1d_momentum_breakout_volume_v1
# Hypothesis: Daily momentum breakout with volume confirmation and weekly trend filter.
# Uses Donchian breakout (20-day high/low) with volume > 1.5x 20-day average.
# Weekly trend filter: price above/below weekly EMA20 for direction.
# Designed to capture momentum moves in both bull and bear markets by trading breakouts
# with institutional volume confirmation. Target: 10-20 trades/year for low fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_momentum_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend filter (weekly EMA20) - load once before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA20 on weekly data
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Daily indicators
    # Donchian channels (20-day high/low)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(avg_volume[i]) or np.isnan(ema20_1w_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema20_1w_aligned[i]
        weekly_downtrend = close[i] < ema20_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below 10-day low or weekly trend reverses
            low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values[i]
            if close[i] < low_10 or not weekly_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 10-day high or weekly trend reverses
            high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values[i]
            if close[i] > high_10 or not weekly_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            if volume_ok:
                # Long breakout: price closes above 20-day high
                if weekly_uptrend and close[i] > high_20[i] and close[i-1] <= high_20[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown: price closes below 20-day low
                elif weekly_downtrend and close[i] < low_20[i] and close[i-1] >= low_20[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals