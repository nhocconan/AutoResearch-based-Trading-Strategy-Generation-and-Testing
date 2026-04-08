#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_volume_v2"
timeframe = "1d"
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
    
    # Get weekly data for trend filter and volume average
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-period EMA on weekly close for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 20-period average volume on weekly
    volume_1w = df_1w['volume'].values
    vol_avg_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_avg_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_1w)
    
    # Volume confirmation: daily volume > 1.5x weekly average volume (scaled to daily)
    # 1 week = 5 trading days, so divide by 5 for per-day comparison
    vol_threshold = vol_avg_1w_aligned / 5.0
    vol_confirm = volume > vol_threshold * 1.5
    
    # Donchian channels on daily data (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup period
    start_idx = max(20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if EMA or volume average not available
        if np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_avg_1w_aligned[i]) or \
           np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or below weekly EMA (trend change)
            if close[i] < donchian_low[i] or close[i] < ema_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or above weekly EMA (trend change)
            if close[i] > donchian_high[i] or close[i] > ema_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with volume confirmation and above weekly EMA
            if high[i] > donchian_high[i-1] and vol_confirm[i] and close[i] > ema_20_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume confirmation and below weekly EMA
            elif low[i] < donchian_low[i-1] and vol_confirm[i] and close[i] < ema_20_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals