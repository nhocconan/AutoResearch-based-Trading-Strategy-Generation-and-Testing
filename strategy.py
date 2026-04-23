#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
- Uses Donchian(20) channels from 1d for breakout signals (proven structure)
- 1w EMA50 defines long-term trend: only long when price > EMA50, short when price < EMA50
- Volume confirmation (> 1.5x 20-period average) filters weak breakouts
- Exit when price crosses 1d EMA34 (adaptive stop) or opposite Donchian channel
- Designed for 1d timeframe targeting 15-25 trades/year (60-100 over 4 years)
- Works in both bull and bear markets by following the 1w EMA50 trend filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d EMA34 for adaptive exit
    ema_34_1d = pd.Series(close).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 50, 34, 20)  # need Donchian(20), 1w EMA50, 1d EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(ema_34_1d[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper AND above 1w EMA50 AND volume spike
            if (close[i] > highest_high[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND below 1w EMA50 AND volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses 1d EMA34 OR touches opposite Donchian channel
            exit_signal = False
            
            if position == 1:
                # Exit long when price < 1d EMA34 OR < Donchian lower
                if close[i] < ema_34_1d[i] or close[i] < lowest_low[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price > 1d EMA34 OR > Donchian upper
                if close[i] > ema_34_1d[i] or close[i] > highest_high[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0