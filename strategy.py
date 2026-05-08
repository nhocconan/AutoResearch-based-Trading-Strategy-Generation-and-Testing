#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_WeeklyDonchianBreakout_TrendVolume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Donchian and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly Donchian channels (20 periods)
    highest_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Weekly volume spike: current volume > 2.0x 20-period average
    vol_ma20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_spike_1w = volume_1w > (2.0 * vol_ma20_1w)
    
    # Align weekly data to daily timeframe
    highest_20_aligned = align_htf_to_ltf(prices, df_1w, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1w, lowest_20)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    volume_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_spike_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high, above weekly EMA50, volume spike
            long_cond = (close[i] > highest_20_aligned[i] and 
                        close[i] > ema50_1w_aligned[i] and
                        volume_spike_1w_aligned[i] > 0.5)
            
            # Short: Price breaks below weekly Donchian low, below weekly EMA50, volume spike
            short_cond = (close[i] < lowest_20_aligned[i] and 
                         close[i] < ema50_1w_aligned[i] and
                         volume_spike_1w_aligned[i] > 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below weekly Donchian low OR below weekly EMA50
            if close[i] < lowest_20_aligned[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above weekly Donchian high OR above weekly EMA50
            if close[i] > highest_20_aligned[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals