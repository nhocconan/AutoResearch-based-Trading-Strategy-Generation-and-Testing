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
    
    # Get weekly data for higher timeframe context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channel (20-period)
    highest_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    highest_20_aligned = align_htf_to_ltf(prices, df_1w, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1w, lowest_20)
    
    # Calculate weekly ATR(14) for volatility filter
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20_aligned[i]) or 
            np.isnan(lowest_20_aligned[i]) or
            np.isnan(atr_14_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Price near weekly Donchian levels
        price_range = highest_20_aligned[i] - lowest_20_aligned[i]
        if price_range <= 0:
            signals[i] = 0.0
            continue
            
        # Normalized position within the range (0 = low, 1 = high)
        pos_in_range = (close[i] - lowest_20_aligned[i]) / price_range
        
        # Volatility filter: avoid extremely high volatility periods
        vol_filter = atr_14_1w_aligned[i] > 0 and atr_14_1w_aligned[i] < np.median(atr_14_1w_aligned[:i+1]) * 3
        
        # Volume filter: above average volume (using weekly average)
        vol_mean_1w = np.mean(volume[max(0, i-7):i+1])  # approximate weekly volume average
        vol_spike = volume[i] > vol_mean_1w * 1.5
        
        # Long conditions: near weekly low + volatility filter + volume spike
        long_condition = (pos_in_range < 0.2 and vol_filter and vol_spike)
        
        # Short conditions: near weekly high + volatility filter + volume spike
        short_condition = (pos_in_range > 0.8 and vol_filter and vol_spike)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: price moves to middle of range
        elif position == 1 and pos_in_range > 0.5:
            signals[i] = 0.0
            position = 0
        elif position == -1 and pos_in_range < 0.5:
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

name = "1d_WeeklyDonchian20_VolumeFilter_Session"
timeframe = "1d"
leverage = 1.0