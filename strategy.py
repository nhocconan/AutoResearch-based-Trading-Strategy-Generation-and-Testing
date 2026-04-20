#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA21 trend filter and volume confirmation
# - Long when price breaks above 20-day high AND 1w EMA21 is rising AND volume > 1.5x 20-day avg volume
# - Short when price breaks below 20-day low AND 1w EMA21 is falling AND volume > 1.5x 20-day avg volume
# - Exit on opposite breakout or when EMA trend reverses
# - Designed for 1d timeframe to capture major trends with few trades (target: 10-25/year)
# - Uses weekly EMA for trend filter to avoid whipsaws in ranging markets

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
    
    # Load 1w data for EMA21 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 20-period Donchian channels on 1d
    highest_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume on 1d
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 21-period EMA on 1w for trend filter
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 1d indicators to 1d timeframe (no alignment needed as we're already on 1d)
    # Align 1w EMA21 to 1d timeframe
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if NaN in indicators
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(avg_volume_20[i]) or np.isnan(ema_21_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume_1d[i]
        donchian_high = highest_high_20[i]
        donchian_low = lowest_low_20[i]
        avg_vol = avg_volume_20[i]
        ema_21 = ema_21_1w_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high AND EMA21 rising AND volume spike
            if (price > donchian_high and 
                ema_21_1w_aligned[i] > ema_21_1w_aligned[i-1] and  # EMA rising
                vol > 1.5 * avg_vol):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low AND EMA21 falling AND volume spike
            elif (price < donchian_low and 
                  ema_21_1w_aligned[i] < ema_21_1w_aligned[i-1] and  # EMA falling
                  vol > 1.5 * avg_vol):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low OR EMA21 turns down
            if (price < donchian_low or 
                ema_21_1w_aligned[i] < ema_21_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high OR EMA21 turns up
            if (price > donchian_high or 
                ema_21_1w_aligned[i] > ema_21_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA21_VolumeFilter"
timeframe = "1d"
leverage = 1.0