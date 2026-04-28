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
    
    # Get 4h data for trend and Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA200 for trend filter
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 4h Donchian(20) channels
    donch_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max()
    donch_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min()
    
    # Align 4h indicators to 4h timeframe (no shift needed as we're already on 4h)
    ema200_4h_aligned = ema200_4h  # Already aligned to 4h
    donch_high_20_aligned = donch_high_20.values
    donch_low_20_aligned = donch_low_20.values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 200  # Wait for EMA200 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_4h_aligned[i]) or 
            np.isnan(donch_high_20_aligned[i]) or
            np.isnan(donch_low_20_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA200
        uptrend = close[i] > ema200_4h_aligned[i]
        downtrend = close[i] < ema200_4h_aligned[i]
        
        # Volume filter: current volume above 1d average
        vol_filter = volume[i] > vol_ma_1d_aligned[i]
        
        # Breakout conditions: price breaks Donchian(20) with volume and trend
        long_breakout = high[i] > donch_high_20_aligned[i]
        short_breakout = low[i] < donch_low_20_aligned[i]
        
        long_entry = long_breakout and uptrend and vol_filter
        short_entry = short_breakout and downtrend and vol_filter
        
        # Exit conditions: price returns to opposite Donchian band or trend reverses
        long_exit = low[i] < donch_low_20_aligned[i] or not uptrend
        short_exit = high[i] > donch_high_20_aligned[i] or not downtrend
        
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
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_4hEMA200_1dVolume_Breakout"
timeframe = "4h"
leverage = 1.0