#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12h EMA(21) for trend filter
    close_12h = df_12h['close'].values
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Get 1d data for 4h Donchian(20) calculation (highest high/lowest low of last 20 4h periods = 10 days)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 10-day high and low for Donchian channel (20 * 4h = 10d)
    high_10d = df_1d['high'].rolling(window=10, min_periods=10).max().values
    low_10d = df_1d['low'].rolling(window=10, min_periods=10).min().values
    high_10d_aligned = align_htf_to_ltf(prices, df_1d, high_10d)
    low_10d_aligned = align_htf_to_ltf(prices, df_1d, low_10d)
    
    # Volume filter: volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 20)  # Wait for EMA and volume
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_12h_aligned[i]) or np.isnan(high_10d_aligned[i]) or 
            np.isnan(low_10d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA(21)
        uptrend = close[i] > ema_21_12h_aligned[i]
        downtrend = close[i] < ema_21_12h_aligned[i]
        
        # Entry conditions: Donchian breakout in trend direction with volume confirmation
        long_breakout = close[i] > high_10d_aligned[i]
        short_breakout = close[i] < low_10d_aligned[i]
        
        long_entry = long_breakout and uptrend and volume_confirm[i]
        short_entry = short_breakout and downtrend and volume_confirm[i]
        
        # Exit conditions: opposite Donchian breakout or trend reversal
        long_exit = (close[i] < low_10d_aligned[i]) or (not uptrend)
        short_exit = (close[i] > high_10d_aligned[i]) or (not downtrend)
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
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

name = "4h_Donchian20_12hEMA21_VolumeConfirm"
timeframe = "4h"
leverage = 1.0