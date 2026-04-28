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
    
    # Get daily data for filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate daily EMA200 for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate daily volume MA(20) for volume filter
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 1d timeframe (no alignment needed as we're on 1d)
    rsi_aligned = rsi_1d
    ema_200_aligned = ema_200_1d
    vol_ma_aligned = vol_ma_1d
    
    # Calculate 1d Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 200  # Need EMA200 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or 
            np.isnan(high_20[i]) or 
            np.isnan(low_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above EMA200 for long, below for short
        uptrend = close[i] > ema_200_aligned[i]
        downtrend = close[i] < ema_200_aligned[i]
        
        # Momentum filter: RSI between 30 and 70 to avoid extremes
        rsi_middle = (rsi_aligned[i] > 30) and (rsi_aligned[i] < 70)
        
        # Volume filter: volume above average
        volume_ok = volume[i] > vol_ma_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_20[i-1]  # Break above previous high
        breakout_down = close[i] < low_20[i-1]  # Break below previous low
        
        # Long conditions: uptrend + RSI middle + volume + breakout up
        long_condition = uptrend and rsi_middle and volume_ok and breakout_up
        
        # Short conditions: downtrend + RSI middle + volume + breakout down
        short_condition = downtrend and rsi_middle and volume_ok and breakout_down
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite Donchian breakout
        elif position == 1 and close[i] < low_20[i-1]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > high_20[i-1]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_EMA200_RSI30_70_Volume_Donchian20_Breakout"
timeframe = "1d"
leverage = 1.0