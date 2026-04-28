#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with 1-week EMA(10) trend filter and volume confirmation.
# The daily timeframe reduces trade frequency to avoid fee drag while the weekly trend filter
# ensures we only trade with the dominant weekly momentum. Volume confirmation filters out
# weak breakouts. This combination has shown robustness in both bull and bear markets by
# capturing strong momentum moves while avoiding chop.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(10) for trend filter
    ema_10 = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_aligned = align_htf_to_ltf(prices, df_1w, ema_10)
    
    # Calculate daily Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-day average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_10_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > avg_volume[i]
        
        # Trend filter: price relative to weekly EMA10
        uptrend = close[i] > ema_10_aligned[i]
        downtrend = close[i] < ema_10_aligned[i]
        
        # Breakout conditions
        long_breakout = close[i] > highest_high[i]
        short_breakout = close[i] < lowest_low[i]
        
        # Entry conditions
        if long_breakout and uptrend and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and downtrend and vol_confirm and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite breakout or trend reversal
        elif position == 1 and (short_breakout or not uptrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (long_breakout or not downtrend):
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

name = "1d_Donchian20_WeeklyEMA10_Volume"
timeframe = "1d"
leverage = 1.0