#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d EMA trend filter and volume confirmation
# Works in bull/bear: trend filter avoids counter-trend trades, volume confirms breakout strength
# Target: 12-37 trades/year (50-150 over 4 years) to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Align daily EMA to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(lookback, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_50_aligned[i]):
            continue
            
        # Volume filter: above average volume
        vol_ma = np.mean(volume[max(0, i-10):i+1])
        if volume[i] < 0.5 * vol_ma:  # Avoid low volume breakouts
            continue
            
        # Long: breakout above Donchian high + above daily EMA50
        if close[i] > highest_high[i] and close[i] > ema_50_aligned[i]:
            if position == 0:
                position = 1
                signals[i] = position_size
            elif position == -1:  # Reverse from short
                position = 1
                signals[i] = position_size
                
        # Short: breakdown below Donchian low + below daily EMA50
        elif close[i] < lowest_low[i] and close[i] < ema_50_aligned[i]:
            if position == 0:
                position = -1
                signals[i] = -position_size
            elif position == 1:  # Reverse from long
                position = -1
                signals[i] = -position_size
                
        # Exit: reverse signal or return to EMA50
        elif position == 1 and close[i] < ema_50_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > ema_50_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian_Breakout_EMA50_Volume"
timeframe = "12h"
leverage = 1.0