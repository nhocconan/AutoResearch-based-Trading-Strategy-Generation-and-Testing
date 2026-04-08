#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_v5"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period EMA on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate 20-period average volume on 1d
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Volume confirmation: 4h volume > 1.5x 1d average volume (scaled to 4h)
    # 1d volume represents 6x 4h periods, so divide by 6 for per-period comparison
    vol_threshold = vol_avg_1d_aligned / 6.0
    vol_confirm = volume > vol_threshold * 1.5
    
    # Donchian channels on 4h data (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup period
    start_idx = max(20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if EMA or volume average not available
        if np.isnan(ema_20_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or \
           np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or below 1d EMA (trend change)
            if close[i] < donchian_low[i] or close[i] < ema_20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or above 1d EMA (trend change)
            if close[i] > donchian_high[i] or close[i] > ema_20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with volume confirmation and above 1d EMA
            if high[i] > donchian_high[i-1] and vol_confirm[i] and close[i] > ema_20_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume confirmation and below 1d EMA
            elif low[i] < donchian_low[i-1] and vol_confirm[i] and close[i] < ema_20_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals