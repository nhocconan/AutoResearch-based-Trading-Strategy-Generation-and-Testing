#!/usr/bin/env python3
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
    
    # Get weekly data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily Donchian channels (20-period)
    highest_high_20d = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20d = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.3x average volume (20-day average)
    vol_ma_20d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > vol_ma_20d * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(highest_high_20d[i]) or
            np.isnan(lowest_low_20d[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from weekly EMA
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > highest_high_20d[i]
        breakout_down = close[i] < lowest_low_20d[i]
        
        # Entry conditions: require trend + breakout + volume confirmation
        long_entry = uptrend and breakout_up and volume_confirm[i]
        short_entry = downtrend and breakout_down and volume_confirm[i]
        
        # Exit conditions: when trend reverses or opposite breakout
        if position == 1:
            exit_condition = not uptrend or breakout_down
        elif position == -1:
            exit_condition = not downtrend or breakout_up
        else:
            exit_condition = False
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif exit_condition and position != 0:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1wDonchian20_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0