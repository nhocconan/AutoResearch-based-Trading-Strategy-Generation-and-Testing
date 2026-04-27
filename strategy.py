#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA(50) trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily Donchian channels (20-period) using previous day's data
    prev_high_max = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    prev_low_min = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to daily timeframe
    donch_high = align_htf_to_ltf(prices, df_1d, prev_high_max)
    donch_low = align_htf_to_ltf(prices, df_1d, prev_low_min)
    
    # Daily volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above Donchian high, above weekly EMA50, volume spike
        if (close[i] > donch_high[i] and 
            close[i] > ema_50_1w_aligned[i] and 
            volume_spike[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below Donchian low, below weekly EMA50, volume spike
        elif (close[i] < donch_low[i] and 
              close[i] < ema_50_1w_aligned[i] and 
              volume_spike[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to opposite Donchian level
        elif position == 1 and close[i] < donch_low[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > donch_high[i]:
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

name = "1d_Donchian20_WeeklyEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0