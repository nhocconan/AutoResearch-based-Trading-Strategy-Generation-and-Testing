#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above 4h Donchian upper band AND 12h EMA50 is rising AND 4h volume > 1.5x 20-period EMA.
# Short when price breaks below 4h Donchian lower band AND 12h EMA50 is falling AND 4h volume > 1.5x 20-period EMA.
# Uses volume breakout for momentum confirmation and EMA trend filter to avoid counter-trend trades.
# Designed for moderate trade frequency (target: 20-40/year) to balance opportunity and fee drag.
# Works in trending markets by following price channels with trend and volume filters.
name = "4h_Donchian20_Volume_EMA50_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50: rising if current > previous, falling if current < previous
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_rising = np.where(ema_50_12h > np.roll(ema_50_12h, 1), 1, 0)
    ema_50_falling = np.where(ema_50_12h < np.roll(ema_50_12h, 1), 1, 0)
    # Handle first value
    ema_50_rising[0] = 0
    ema_50_falling[0] = 0
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_50_falling)
    
    # 4h Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume confirmation: volume > 1.5x 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long condition: break above Donchian upper, EMA50 rising, volume confirmation
            long_condition = (close[i] > highest_high_20[i]) and ema_50_rising_aligned[i] and vol_confirm[i]
            # Short condition: break below Donchian lower, EMA50 falling, volume confirmation
            short_condition = (close[i] < lowest_low_20[i]) and ema_50_falling_aligned[i] and vol_confirm[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Donchian lower band
            if close[i] < lowest_low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Donchian upper band
            if close[i] > highest_high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals