#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend + volume confirmation
# - Long when price breaks above Donchian upper band (20-day high) and 1w EMA50 > prior 1w EMA50
# - Short when price breaks below Donchian lower band (20-day low) and 1w EMA50 < prior 1w EMA50
# - Volume must be > 1.5x 20-day average volume
# - Designed for 1d timeframe with selective entries to avoid overtrading
# - Target: 7-25 trades per year per symbol (30-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Donchian and volume calculations
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian channels (20-period)
    highest_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate average volume (20-period)
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Load 1w data for EMA50 trend
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after Donchian and volume warmup
        # Skip if NaN in indicators
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(avg_volume_20[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        volume = volume_1d[i]
        donchian_upper = highest_high_20[i]
        donchian_lower = lowest_low_20[i]
        avg_volume = avg_volume_20[i]
        ema_50 = ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_ok = volume > 1.5 * avg_volume
        
        if position == 0:
            # Long entry: price breaks above Donchian upper + EMA50 rising + volume confirmation
            if price > donchian_upper and ema_50 > ema_50_1w_aligned[i-1] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower + EMA50 falling + volume confirmation
            elif price < donchian_lower and ema_50 < ema_50_1w_aligned[i-1] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian lower or EMA50 turns down
            if price < donchian_lower or ema_50 < ema_50_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian upper or EMA50 turns up
            if price > donchian_upper or ema_50 > ema_50_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0