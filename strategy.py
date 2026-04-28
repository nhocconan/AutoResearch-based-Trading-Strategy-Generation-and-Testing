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
    
    # Get 12h data for HTF trend and pivot levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h EMA(50) for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 12h volume moving average (20-period)
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h volume MA to 4h timeframe
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Calculate 12h volume spike (current volume > 2x 20-period MA)
    vol_spike_12h = volume_12h > (2.0 * vol_ma_20_12h)
    vol_spike_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h.astype(float))
    
    # Calculate 12h Pivot Points (standard floor trader method)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    r1_12h = 2.0 * pivot_12h - low_12h
    s1_12h = 2.0 * pivot_12h - high_12h
    
    # Align 12h pivot levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema50_12h_aligned[i]
        downtrend = close[i] < ema50_12h_aligned[i]
        
        # 12h pivot levels
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        
        # Volume spike confirmation from 12h timeframe
        vol_spike = vol_spike_aligned[i] > 0.5
        
        # Entry conditions: 
        # Long: Price breaks above 12h R1 with volume spike and uptrend
        # Short: Price breaks below 12h S1 with volume spike and downtrend
        long_entry = (close[i] > r1) and vol_spike and uptrend
        short_entry = (close[i] < s1) and vol_spike and downtrend
        
        # Exit conditions: 
        # Long exit: price returns below 12h pivot or trend reversal
        # Short exit: price returns above 12h pivot or trend reversal
        pivot_val = (r1 + s1) / 2.0  # Pivot level
        
        long_exit = (close[i] < pivot_val) or (not uptrend)
        short_exit = (close[i] > pivot_val) or (not downtrend)
        
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

name = "4h_12hPivot_R1S1_Breakout_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0