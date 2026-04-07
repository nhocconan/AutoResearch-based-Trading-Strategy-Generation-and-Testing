#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Donchian breakout with weekly trend filter and volume confirmation
# Long when price breaks above weekly Donchian high + daily volume confirmation
# Short when price breaks below weekly Donchian low + daily volume confirmation
# Uses weekly trend to filter direction: only long in weekly uptrend, only short in weekly downtrend
# Target: 7-25 trades/year, low frequency to minimize fee drag
name = "1d_donchian20_1w_trend_volume_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period high/low)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_highest = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    weekly_lowest = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    weekly_highest_aligned = align_htf_to_ltf(prices, df_1w, weekly_highest)
    weekly_lowest_aligned = align_htf_to_ltf(prices, df_1w, weekly_lowest)
    
    # Calculate weekly trend: slope of 20-period EMA of weekly close
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, adjust=False).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    # Trend: 1 if EMA rising, -1 if falling, 0 if flat
    weekly_trend = np.zeros(len(weekly_ema_aligned))
    weekly_trend[1:] = np.where(weekly_ema_aligned[1:] > weekly_ema_aligned[:-1], 1, -1)
    
    # Calculate daily volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(weekly_highest_aligned[i]) or np.isnan(weekly_lowest_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(weekly_trend[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current daily volume > 20-day average volume
        vol_confirm = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price touches weekly lower band OR trend turns down
            if close[i] <= weekly_lowest_aligned[i] or weekly_trend[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price touches weekly upper band OR trend turns up
            if close[i] >= weekly_highest_aligned[i] or weekly_trend[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price breaks above weekly upper band + volume confirmation + weekly uptrend
            if close[i] > weekly_highest_aligned[i] and vol_confirm and weekly_trend[i] == 1:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below weekly lower band + volume confirmation + weekly downtrend
            elif close[i] < weekly_lowest_aligned[i] and vol_confirm and weekly_trend[i] == -1:
                position = -1
                signals[i] = -0.25
    
    return signals