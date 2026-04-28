#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h Donchian channels (18-period) - tighter for fewer trades
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 18:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    highest_high_6h = pd.Series(high_6h).rolling(window=18, min_periods=18).max().values
    lowest_low_6h = pd.Series(low_6h).rolling(window=18, min_periods=18).min().values
    highest_high_6h_aligned = align_htf_to_ltf(prices, df_6h, highest_high_6h)
    lowest_low_6h_aligned = align_htf_to_ltf(prices, df_6h, lowest_low_6h)
    
    # Volume confirmation: current volume > 1.8x average volume (6h average)
    vol_ma_6h = pd.Series(volume).rolling(window=18, min_periods=18).mean().values
    volume_confirm = volume > vol_ma_6h * 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 18, 18)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(highest_high_6h_aligned[i]) or
            np.isnan(lowest_low_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > highest_high_6h_aligned[i]
        breakout_down = close[i] < lowest_low_6h_aligned[i]
        
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

name = "6h_Donchian18_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0