#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly EMA(10) for trend filter
    close_1w = df_1w['close'].values
    ema10_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 10:
        ema10_1w[9] = np.mean(close_1w[0:10])
        for i in range(10, len(close_1w)):
            ema10_1w[i] = (close_1w[i] * 2 + ema10_1w[i-1] * 9) / 10
    
    # Align weekly EMA to daily timeframe
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    # Load daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-day Donchian channels
    upper_dc_1d = np.full_like(high_1d, np.nan)
    lower_dc_1d = np.full_like(low_1d, np.nan)
    
    for i in range(19, len(high_1d)):
        upper_dc_1d[i] = np.max(high_1d[i-19:i+1])
        lower_dc_1d[i] = np.min(low_1d[i-19:i+1])
    
    # Align daily Donchian to daily timeframe (no change, but required for consistency)
    upper_dc_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_dc_1d)
    lower_dc_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_dc_1d)
    
    # Calculate daily ATR for volatility filter
    if len(high_1d) < 2:
        return np.zeros(n)
    
    tr = np.zeros_like(high_1d)
    for i in range(1, len(high_1d)):
        tr[i] = max(high_1d[i] - low_1d[i],
                   abs(high_1d[i] - high_1d[i-1]),
                   abs(low_1d[i] - low_1d[i-1]))
    
    atr_1d = np.full_like(high_1d, np.nan)
    if len(high_1d) >= 14:
        atr_1d[13] = np.mean(tr[1:14])
        for i in range(14, len(high_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align daily ATR to daily timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # Position size: 25% of capital
    
    for i in range(30, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema10_1w_aligned[i]) or 
            np.isnan(upper_dc_1d_aligned[i]) or
            np.isnan(lower_dc_1d_aligned[i]) or
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current day volume vs 20-day average
        vol_ma_20 = np.full_like(volume, np.nan)
        for j in range(19, len(volume)):
            vol_ma_20[j] = np.mean(volume[j-19:j+1])
        
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with weekly uptrend and volume surge
            if (close[i] > upper_dc_1d_aligned[i] and 
                close[i] > ema10_1w_aligned[i] and
                volume_ratio > 2.0):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower Donchian with weekly downtrend and volume surge
            elif (close[i] < lower_dc_1d_aligned[i] and 
                  close[i] < ema10_1w_aligned[i] and
                  volume_ratio > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls below lower Donchian or weekly trend turns down
            if (close[i] < lower_dc_1d_aligned[i] or
                close[i] < ema10_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises above upper Donchian or weekly trend turns up
            if (close[i] > upper_dc_1d_aligned[i] or
                close[i] > ema10_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_WeeklyEMA10_Donchian20_Volume"
timeframe = "1d"
leverage = 1.0