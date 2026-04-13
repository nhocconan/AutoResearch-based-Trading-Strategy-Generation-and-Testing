#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h strategy using 4h Donchian breakout + volume confirmation + session filter (08-20 UTC)
    # Long: price > 4h Donchian high(20) AND volume > 1.5x 20-period avg AND hour in [8,20) UTC
    # Short: price < 4h Donchian low(20) AND volume > 1.5x 20-period avg AND hour in [8,20) UTC
    # Exit: opposite Donchian breakout or volume dry-up or outside session
    # Uses 4h for signal direction (low frequency), 1h only for entry timing and exit.
    # Discrete position sizing (0.20) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours < 20)
    
    # Get 4h data for Donchian channels (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period high/low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian high(20) and low(20) on 4h
    donchian_high_4h = np.full(len(df_4h), np.nan)
    donchian_low_4h = np.full(len(df_4h), np.nan)
    
    for i in range(20, len(df_4h)):
        donchian_high_4h[i] = np.max(high_4h[i-20:i])
        donchian_low_4h[i] = np.min(low_4h[i-20:i])
    
    # Align 4h Donchian levels to 1h (waits for completed 4h bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    
    # Volume confirmation: >1.5x 20-period average on 1h
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        session_ok = in_session[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: Donchian breakout + volume + session
        long_entry = (close[i] > donchian_high_aligned[i]) and vol_confirm and session_ok
        short_entry = (close[i] < donchian_low_aligned[i]) and vol_confirm and session_ok
        
        # Exit logic: opposite breakout or volume dry-up or outside session
        long_exit = (close[i] < donchian_low_aligned[i]) or not vol_confirm or not session_ok
        short_exit = (close[i] > donchian_high_aligned[i]) or not vol_confirm or not session_ok
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_donchian_breakout_volume_session_v1"
timeframe = "1h"
leverage = 1.0