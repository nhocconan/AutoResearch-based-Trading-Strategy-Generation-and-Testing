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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for Donchian and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Weekly EMA(21) for trend filter
    ema21_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Daily Donchian channels (20)
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily volume SMA(20) for confirmation
    vol_sma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly trend to daily
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Align daily indicators to 1d timeframe (self-alignment)
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    vol_sma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20)
    
    # Align to 1d timeframe (since we're using 1d timeframe)
    # For 1d timeframe, we need to align to 1d bars
    upper_aligned = upper_20_aligned
    lower_aligned = lower_20_aligned
    vol_sma_aligned = vol_sma_20_aligned
    
    # Calculate signal only at daily close (we'll use close price)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema21_1w_aligned[i]) or 
            np.isnan(upper_aligned[i]) or
            np.isnan(lower_aligned[i]) or
            np.isnan(vol_sma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA21
        uptrend = close[i] > ema21_1w_aligned[i]
        downtrend = close[i] < ema21_1w_aligned[i]
        
        # Volume filter: current volume above average
        vol_filter = volume[i] > vol_sma_aligned[i]
        
        # Breakout conditions
        long_breakout = close[i] > upper_aligned[i]
        short_breakout = close[i] < lower_aligned[i]
        
        long_entry = long_breakout and uptrend and vol_filter
        short_entry = short_breakout and downtrend and vol_filter
        
        # Exit conditions: return to middle of channel or trend reversal
        mid_channel = (upper_aligned[i] + lower_aligned[i]) / 2.0
        long_exit = close[i] < mid_channel[i] or not uptrend
        short_exit = close[i] > mid_channel[i] or not downtrend
        
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

name = "1d_Donchian20_WeeklyEMA21_Trend_Volume"
timeframe = "1d"
leverage = 1.0