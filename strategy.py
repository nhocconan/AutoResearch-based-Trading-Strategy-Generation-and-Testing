#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Choppiness_Filtered_Donchian_Breakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend direction
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 4-hour Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_channel = high_roll
    lower_channel = low_roll
    
    # Shift to get previous bar's channels (no look-ahead)
    upper_channel_prev = np.roll(upper_channel, 1)
    lower_channel_prev = np.roll(lower_channel, 1)
    upper_channel_prev[0] = np.nan
    lower_channel_prev[0] = np.nan
    
    # Calculate 4-hour Choppiness Index (14-period)
    def calculate_choppiness(high, low, close, window):
        atr_list = []
        for i in range(len(high)):
            if i == 0:
                tr = high[0] - low[0]
            else:
                tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr_list.append(tr)
        
        atr_array = np.array(atr_list)
        atr_sum = pd.Series(atr_array).rolling(window=window, min_periods=window).sum().values
        
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        
        chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(window)
        return chop
    
    chop = calculate_choppiness(high, low, close, 14)
    
    # Volume spike detection: current volume > 2 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(upper_channel_prev[i]) or 
            np.isnan(lower_channel_prev[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema34_1d_aligned[i]
        chop_val = chop[i]
        vol_spike = volume_spike[i]
        
        # Only trade in trending markets (Choppiness < 38.2)
        is_trending = chop_val < 38.2
        
        if position == 0:
            # Enter long: price breaks above upper channel with volume spike, above 1d EMA, in trending market
            if (close[i] > upper_channel_prev[i] and vol_spike and 
                close[i] > ema_val and is_trending):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower channel with volume spike, below 1d EMA, in trending market
            elif (close[i] < lower_channel_prev[i] and vol_spike and 
                  close[i] < ema_val and is_trending):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower channel OR below 1d EMA OR chop becomes too high
            if (close[i] < lower_channel_prev[i] or close[i] < ema_val or chop_val > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper channel OR above 1d EMA OR chop becomes too high
            if (close[i] > upper_channel_prev[i] or close[i] > ema_val or chop_val > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals