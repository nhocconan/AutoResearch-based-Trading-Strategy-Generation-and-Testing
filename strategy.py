#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA 34 for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d volume spike: volume > 2x 20-period average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (vol_ma_1d * 2.0)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate 6h Donchian channels (20-period)
    # We'll use the 6h data from prices directly since it's our timeframe
    # But we need to calculate rolling max/min with proper lookback
    highest_high_6h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_6h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(highest_high_6h[i]) or 
            np.isnan(lowest_low_6h[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: 1d volume spike
        vol_spike = volume_spike_1d_aligned[i]
        
        # Long conditions: price breaks above 6h Donchian high + above 1d EMA + volume spike
        long_breakout = (close[i] > highest_high_6h[i-1] and 
                        price_above_ema and 
                        vol_spike)
        
        # Short conditions: price breaks below 6h Donchian low + below 1d EMA + volume spike
        short_breakout = (close[i] < lowest_low_6h[i-1] and 
                         price_below_ema and 
                         vol_spike)
        
        if long_breakout:
            signals[i] = 0.25
            position = 1
        elif short_breakout:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite 6h Donchian breakout
        elif position == 1 and close[i] < lowest_low_6h[i-1]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > highest_high_6h[i-1]:
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

name = "6h_Donchian20_Breakout_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0