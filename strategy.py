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
    
    # Get daily data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA50 for trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily ATR(14) for volatility
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily volume moving average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 12h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 12-hour Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(atr_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or 
            np.isnan(high_20[i]) or 
            np.isnan(low_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above EMA50 for long, below for short
        uptrend = close[i] > ema50_aligned[i]
        downtrend = close[i] < ema50_aligned[i]
        
        # Volatility filter: sufficient ATR
        vol_ok = atr_aligned[i] > (vol_ma_aligned[i] * 0.1)
        
        # Volume filter: current volume above 12h average
        vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_ma_12h[i]):
            signals[i] = 0.0
            continue
        vol_spike = volume[i] > (vol_ma_12h[i] * 1.5)
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_20[i-1]  # Break above previous high
        breakout_down = close[i] < low_20[i-1]  # Break below previous low
        
        # Long conditions: uptrend + volatility + volume spike + breakout up
        long_condition = uptrend and vol_ok and vol_spike and breakout_up
        
        # Short conditions: downtrend + volatility + volume spike + breakout down
        short_condition = downtrend and vol_ok and vol_spike and breakout_down
        
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

name = "12h_Donchian20_Breakout_EMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0