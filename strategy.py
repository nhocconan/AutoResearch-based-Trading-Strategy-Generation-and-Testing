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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1w data for Donchian channels (10-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    highest_high_1w = pd.Series(high_1w).rolling(window=10, min_periods=10).max().values
    lowest_low_1w = pd.Series(low_1w).rolling(window=10, min_periods=10).min().values
    highest_high_1w_aligned = align_htf_to_ltf(prices, df_1w, highest_high_1w)
    lowest_low_1w_aligned = align_htf_to_ltf(prices, df_1w, lowest_low_1w)
    
    # Volume confirmation: current volume > 1.5x average volume (1d average)
    vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > vol_ma_1d * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(highest_high_1w_aligned[i]) or
            np.isnan(lowest_low_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > highest_high_1w_aligned[i]
        breakout_down = close[i] < lowest_low_1w_aligned[i]
        
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
            signals[i] = 0.30
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.30
            position = -1
        elif exit_condition and position != 0:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1wDonchian10_1dEMA34_VolumeConfirm"
timeframe = "1d"
leverage = 1.0