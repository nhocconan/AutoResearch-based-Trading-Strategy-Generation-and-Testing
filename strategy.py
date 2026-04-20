#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA21 trend + Volume confirmation
# - Long when price breaks above 20-day high AND price > 1w EMA21 AND volume > 1.5x 20-day avg volume
# - Short when price breaks below 20-day low AND price < 1w EMA21 AND volume > 1.5x 20-day avg volume
# - Uses daily price structure with weekly trend filter to avoid counter-trend trades
# - Volume filter ensures breakouts have conviction
# - Designed for 1d timeframe with selective entries to avoid overtrading
# - Target: 7-25 trades per year per symbol (30-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Donchian channels and volume average
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian channels (20-period)
    highest_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-day average volume
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Load 1w data for EMA21 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if NaN in indicators
        if np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or \
           np.isnan(avg_volume_20[i]) or np.isnan(ema_21_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        volume = volume_1d[i]
        
        if position == 0:
            # Long entry: price > 20-day high AND price > 1w EMA21 AND volume > 1.5x avg volume
            if price > highest_high_20[i] and price > ema_21_1w_aligned[i] and volume > 1.5 * avg_volume_20[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price < 20-day low AND price < 1w EMA21 AND volume > 1.5x avg volume
            elif price < lowest_low_20[i] and price < ema_21_1w_aligned[i] and volume > 1.5 * avg_volume_20[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < 20-day low OR price < 1w EMA21
            if price < lowest_low_20[i] or price < ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > 20-day high OR price > 1w EMA21
            if price > highest_high_20[i] or price > ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA21_VolumeFilter"
timeframe = "1d"
leverage = 1.0