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
    
    # Load daily data (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels (daily)
    if len(high_1d) < 20:
        return np.zeros(n)
    
    # Upper band: highest high over last 20 periods
    upper_20 = np.full_like(high_1d, np.nan)
    for i in range(19, len(high_1d)):
        upper_20[i] = np.max(high_1d[i-19:i+1])
    
    # Lower band: lowest low over last 20 periods
    lower_20 = np.full_like(low_1d, np.nan)
    for i in range(19, len(low_1d)):
        lower_20[i] = np.min(low_1d[i-19:i+1])
    
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Calculate 50-day SMA for trend filter (daily)
    if len(close_1d) < 50:
        return np.zeros(n)
    
    sma50_1d = np.full_like(close_1d, np.nan)
    for i in range(49, len(close_1d)):
        sma50_1d[i] = np.mean(close_1d[i-49:i+1])
    
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    # Calculate 14-day RSI for momentum (daily)
    if len(close_1d) < 14:
        return np.zeros(n)
    
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= 14:
        avg_gain[13] = np.mean(gain[1:14])
        avg_loss[13] = np.mean(loss[1:14])
        for i in range(14, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full_like(close_1d, np.nan)
    rsi14 = np.full_like(close_1d, np.nan)
    for i in range(13, len(close_1d)):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi14[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi14[i] = 100 if avg_gain[i] > 0 else 0
    
    rsi14_aligned = align_htf_to_ltf(prices, df_1d, rsi14)
    
    # Calculate volume moving average (20-period)
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # Reduced to 25% to lower risk and trade frequency
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or 
            np.isnan(sma50_1d_aligned[i]) or 
            np.isnan(rsi14_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above upper Donchian + price above SMA50 + RSI > 55 + volume surge
            if (close[i] > upper_20_aligned[i] and
                close[i] > sma50_1d_aligned[i] and
                rsi14_aligned[i] > 55 and
                volume_ratio > 2.5):  # Increased volume threshold to reduce trades
                position = 1
                signals[i] = position_size
            # Short: Price breaks below lower Donchian + price below SMA50 + RSI < 45 + volume surge
            elif (close[i] < lower_20_aligned[i] and
                  close[i] < sma50_1d_aligned[i] and
                  rsi14_aligned[i] < 45 and
                  volume_ratio > 2.5):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price breaks below lower Donchian OR RSI < 40
            if (close[i] < lower_20_aligned[i] or 
                rsi14_aligned[i] < 40):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price breaks above upper Donchian OR RSI > 60
            if (close[i] > upper_20_aligned[i] or 
                rsi14_aligned[i] > 60):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Donchian20_SMA50_RSI14_Volume_v2"
timeframe = "12h"
leverage = 1.0