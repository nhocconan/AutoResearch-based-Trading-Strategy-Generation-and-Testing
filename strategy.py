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
    
    # Get daily data for indicators (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 10-period Donchian channels on daily (upper and lower bands)
    upper_channel = np.full_like(close_1d, np.nan)
    lower_channel = np.full_like(close_1d, np.nan)
    
    for i in range(9, len(close_1d)):
        upper_channel[i] = np.max(high_1d[i-9:i+1])
        lower_channel[i] = np.min(low_1d[i-9:i+1])
    
    # Calculate 20-period EMA on daily for trend filter
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all daily data to 12h timeframe (primary)
    upper_channel_12h = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_channel_12h = align_htf_to_ltf(prices, df_1d, lower_channel)
    ema_20_12h = align_htf_to_ltf(prices, df_1d, ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(9, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_channel_12h[i]) or np.isnan(lower_channel_12h[i]) or 
            np.isnan(ema_20_12h[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA
        uptrend = close[i] > ema_20_12h[i]
        downtrend = close[i] < ema_20_12h[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian channel with uptrend
            if close[i] > upper_channel_12h[i] and uptrend:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below lower Donchian channel with downtrend
            elif close[i] < lower_channel_12h[i] and downtrend:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below lower Donchian channel OR trend reverses
            if (close[i] < lower_channel_12h[i]) or (not uptrend):
                signals[i] = 0.0  # flat
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses above upper Donchian channel OR trend reverses
            if (close[i] > upper_channel_12h[i]) or (not downtrend):
                signals[i] = 0.0  # flat
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "12h_Donchian10_1dEMA20_v1"
timeframe = "12h"
leverage = 1.0