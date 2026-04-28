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
    
    # Get daily data for ATR and close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align daily ATR to 4h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    
    # Calculate 4-period ATR-based channel (similar to Donchian but ATR-based)
    # Upper channel: highest high over last 4 periods + ATR
    # Lower channel: lowest low over last 4 periods - ATR
    high_roll_max = pd.Series(high).rolling(window=4, min_periods=4).max().values
    low_roll_min = pd.Series(low).rolling(window=4, min_periods=4).min().values
    
    upper_channel = high_roll_max + atr_aligned
    lower_channel = low_roll_min - atr_aligned
    
    # Calculate average volume over 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_aligned[i]) or 
            np.isnan(upper_channel[i]) or
            np.isnan(lower_channel[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR must be above its 20-period average to avoid low-vol chop
        atr_ma = pd.Series(atr_aligned).rolling(window=20, min_periods=20).mean().values
        vol_filter = atr_aligned[i] > atr_ma[i]
        
        # Volume filter: current volume above average
        vol_filter = vol_filter and (volume[i] > vol_ma[i])
        
        # Entry conditions: breakout of ATR-based channel with volume and volatility filter
        long_entry = (close[i] > upper_channel[i]) and vol_filter
        short_entry = (close[i] < lower_channel[i]) and vol_filter
        
        # Exit conditions: return to middle of channel or volatility drops
        mid_channel = (upper_channel[i] + lower_channel[i]) / 2.0
        long_exit = (close[i] < mid_channel[i]) or (atr_aligned[i] < atr_ma[i])
        short_exit = (close[i] > mid_channel[i]) or (atr_aligned[i] < atr_ma[i])
        
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

name = "4h_ATR_Breakout_Channel_Volume_VolFilter"
timeframe = "4h"
leverage = 1.0