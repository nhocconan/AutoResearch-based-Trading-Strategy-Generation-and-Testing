#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d EMA(50) trend filter + volume confirmation (1.5x 24-bar MA)
- Donchian breakout captures momentum in direction of higher timeframe trend
- 1d EMA(50) ensures alignment with daily trend (bull/bear agnostic)
- Volume confirmation filters false breakouts
- Designed for 4h timeframe targeting 20-50 trades/year (80-200 over 4 years) to minimize fee drag
- Uses discrete position sizing (0.25) to reduce churn
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels (20-period) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: > 1.5x 24-period average (24 * 4h = 4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(60, lookback, 24)  # EMA1d, Donchian, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout signals with trend filter and volume confirmation
        # Long: price breaks above Donchian high + uptrend + volume spike
        # Short: price breaks below Donchian low + downtrend + volume spike
        long_signal = (close[i] > highest_high[i] and 
                      close[i] > ema_50_1d_aligned[i] and
                      volume[i] > 1.5 * vol_ma[i])
        
        short_signal = (close[i] < lowest_low[i] and 
                       close[i] < ema_50_1d_aligned[i] and
                       volume[i] > 1.5 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend reversal or opposite Donchian break
            exit_signal = False
            
            if position == 1:
                # Exit long: trend reversal or price breaks below Donchian low
                if (close[i] < ema_50_1d_aligned[i] or 
                    close[i] < lowest_low[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: trend reversal or price breaks above Donchian high
                if (close[i] > ema_50_1d_aligned[i] or 
                    close[i] > highest_high[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0