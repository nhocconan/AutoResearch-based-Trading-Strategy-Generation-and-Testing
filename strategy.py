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
    
    # Get 12h data for HTF trend filter and volume spike
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # 12h EMA(50) for HTF trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 12h volume spike filter: current 12h volume > 1.5x 20-period average
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume_12h > (vol_ma_20_12h * 1.5)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h.astype(float))
    
    # 4h Donchian(20) breakout levels
    donchian_window = 20
    upper_4h = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_4h = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_window, 50)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_spike_12h_aligned[i]) or
            np.isnan(upper_4h[i]) or 
            np.isnan(lower_4h[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 12h EMA(50)
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Volume spike filter from 12h
        volume_filter = vol_spike_12h_aligned[i] > 0.5  # True if spike
        
        # Entry conditions: Donchian breakout with HTF trend and volume spike
        long_breakout = close[i] > upper_4h[i]
        short_breakout = close[i] < lower_4h[i]
        
        long_entry = uptrend and long_breakout and volume_filter
        short_entry = downtrend and short_breakout and volume_filter
        
        # Exit conditions: Close back inside Donchian channel
        long_exit = close[i] < upper_4h[i]  # Exit long when price falls below upper band
        short_exit = close[i] > lower_4h[i]  # Exit short when price rises above lower band
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0