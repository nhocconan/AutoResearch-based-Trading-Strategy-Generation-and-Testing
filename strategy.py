#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeFilter_v1
Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation. 
In bull/bear markets, breakouts of 20-day high/low with volume spike and aligned weekly trend capture sustained moves. 
Weekly trend filter avoids counter-trend breakouts. Volume confirmation reduces false breakouts. 
Discrete position sizing (0.25) limits fee churn. Targets 20-50 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    weekly_trend = np.where(close > ema_50_1w_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian, 20 for volume MA, 50 for weekly EMA)
    start_idx = max(lookback, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout logic with filters
        long_breakout = close[i] > highest_high[i-1]  # Break above 20-day high
        short_breakout = close[i] < lowest_low[i-1]   # Break below 20-day low
        
        if long_breakout and volume_spike[i] and weekly_trend[i] == 1:
            # Long breakout with volume and weekly uptrend
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        elif short_breakout and volume_spike[i] and weekly_trend[i] == -1:
            # Short breakout with volume and weekly downtrend
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        else:
            # Exit conditions: price returns to mid-channel or opposite breakout
            mid_channel = (highest_high[i] + lowest_low[i]) / 2
            if position == 1 and close[i] < mid_channel:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] > mid_channel:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0