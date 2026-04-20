#!/usr/bin/env python3
# 4h_12h_Donchian_Breakout_Volume_Trend_v1
# Hypothesis: On 4h timeframe, trade Donchian(20) breakouts with volume confirmation and 12h EMA21 trend filter.
# In trending markets (price above/below EMA21), breakouts continue the trend; in ranging markets, avoid false breakouts.
# Targets 20-50 trades/year by requiring confluence of breakout, volume spike, and trend filter.
# Works in both bull and bear markets by filtering breakouts with trend direction.

name = "4h_12h_Donchian_Breakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA21 for trend filter
    close_12h = df_12h['close'].values
    ema_21 = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_12h, ema_21)
    
    # Calculate Donchian channels (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_21_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price > Donchian high + volume spike + above EMA21
            if (close[i] > highest_high[i] and 
                volume[i] > 2.0 * volume_ma[i] and
                close[i] > ema_21_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < Donchian low + volume spike + below EMA21
            elif (close[i] < lowest_low[i] and 
                  volume[i] > 2.0 * volume_ma[i] and
                  close[i] < ema_21_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < Donchian low or trend flip
            if close[i] < lowest_low[i] or close[i] < ema_21_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > Donchian high or trend flip
            if close[i] > highest_high[i] or close[i] > ema_21_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals