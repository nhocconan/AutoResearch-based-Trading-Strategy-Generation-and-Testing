#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d EMA20 trend filter and volume confirmation
# Donchian captures breakouts in trending markets
# 1d EMA20 provides higher timeframe bias to avoid counter-trend trades
# Volume confirmation filters weak breakouts and confirms strength
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries
name = "12h_Donchian20_1dEMA20_Volume"
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
    
    # 1d EMA20 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Donchian channel on 12h
    period = 20
    
    # Highest high and lowest low over period
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(period-1, n):
        highest_high[i] = np.max(high[i-(period-1):i+1])
        lowest_low[i] = np.min(low[i-(period-1):i+1])
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period-1, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close breaks above upper Donchian + above 1d EMA20 + volume confirmation
            if (close[i] > highest_high[i-1] and 
                close[i] > ema_20_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below lower Donchian + below 1d EMA20 + volume confirmation
            elif (close[i] < lowest_low[i-1] and 
                  close[i] < ema_20_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if close breaks below lower Donchian or below 1d EMA20
            if (close[i] < lowest_low[i-1]) or (close[i] < ema_20_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if close breaks above upper Donchian or above 1d EMA20
            if (close[i] > highest_high[i-1]) or (close[i] > ema_20_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals