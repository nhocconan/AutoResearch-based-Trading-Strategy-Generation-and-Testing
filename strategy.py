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
    
    # Get 1d data for HTF calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels on 1d
    high_20 = np.full(len(close_1d), np.nan)
    low_20 = np.full(len(close_1d), np.nan)
    for i in range(20, len(close_1d)):
        high_20[i] = np.max(high_1d[i-20:i])
        low_20[i] = np.min(low_1d[i-20:i])
    
    # Calculate ATR(14) on 1d for volatility filter
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr_14[i] = np.mean(tr[1:15])
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate 20-period average volume on 1d
    vol_ma_20 = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_20[i] = np.mean(volume_1d[i-20:i])
    
    # Align indicators to 6h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        long_breakout = close[i] > high_20_aligned[i]
        short_breakout = close[i] < low_20_aligned[i]
        
        # Volatility filter: current 6h ATR > 1.5 * daily ATR
        # Calculate current 6h ATR (simplified as high-low)
        current_atr_6h = high[i] - low[i]
        vol_filter = current_atr_6h > (1.5 * atr_14_aligned[i])
        
        # Volume filter: current volume > 1.2 * daily average volume
        vol_filter = vol_filter and (volume[i] > (1.2 * vol_ma_20_aligned[i]))
        
        # Entry conditions: breakout with volume and volatility confirmation
        long_entry = long_breakout and vol_filter
        short_entry = short_breakout and vol_filter
        
        # Exit conditions: opposite breakout
        exit_long = position == 1 and short_breakout
        exit_short = position == -1 and long_breakout
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_donchian_breakout_vol_filter"
timeframe = "6h"
leverage = 1.0