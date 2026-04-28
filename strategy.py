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
    
    # Get daily data for 200-day EMA trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA(200) for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate daily volume SMA(20) for volume confirmation
    vol_sma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 12h timeframe
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    vol_sma20_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    # Calculate average volume over 20 periods on 12h chart
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_aligned[i]) or 
            np.isnan(vol_sma20_aligned[i]) or
            np.isnan(vol_ma_12h[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA200
        uptrend = close[i] > ema200_aligned[i]
        downtrend = close[i] < ema200_aligned[i]
        
        # Volume filter: current volume above average (both 12h and daily)
        vol_filter_12h = volume[i] > vol_ma_12h[i]
        vol_filter_1d = volume_1d[i // 48] > vol_sma20_aligned[i] if i >= 48 else False  # 48 = 12h bars per day
        
        # Simple breakout: price breaks above/below EMA200 with volume confirmation
        long_entry = uptrend and vol_filter_12h and vol_filter_1d
        short_entry = downtrend and vol_filter_12h and vol_filter_1d
        
        # Exit conditions: trend reversal or volume drying up
        long_exit = not uptrend or not vol_filter_12h
        short_exit = not downtrend or not vol_filter_12h
        
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

name = "12h_EMA200_Volume_Trend_Filter"
timeframe = "12h"
leverage = 1.0