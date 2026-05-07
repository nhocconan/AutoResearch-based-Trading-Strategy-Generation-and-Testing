#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1-week trend filter and volume confirmation.
# Long when price breaks above Donchian high(20) AND weekly EMA50 is rising AND volume > 2x 20-period average.
# Short when price breaks below Donchian low(20) AND weekly EMA50 is falling AND volume > 2x 20-period average.
# Exit when price crosses back below/above Donchian median (midpoint) or volume drops below average.
# Designed for 6h timeframe with moderate trade frequency (target: 15-30/year) to avoid fee drag.
# Uses weekly EMA50 for trend filter to avoid counter-trend trades in strong trends.
# Volume filter ensures participation and avoids low-conviction moves.
name = "6h_Donchian_20_1wEMA50_VolumeFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_roll + low_roll) / 2.0
    
    # Weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Weekly EMA50 direction (rising/falling)
    ema50_1w_rising = np.zeros_like(ema50_1w_aligned, dtype=bool)
    ema50_1w_falling = np.zeros_like(ema50_1w_aligned, dtype=bool)
    ema50_1w_rising[1:] = ema50_1w_aligned[1:] > ema50_1w_aligned[:-1]
    ema50_1w_falling[1:] = ema50_1w_aligned[1:] < ema50_1w_aligned[:-1]
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(ema50_1w_rising[i]) or np.isnan(ema50_1w_falling[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Donchian high, weekly EMA50 rising, volume filter
            long_cond = (close[i] > high_roll[i]) and ema50_1w_rising[i] and volume_filter[i]
            # Short conditions: break below Donchian low, weekly EMA50 falling, volume filter
            short_cond = (close[i] < low_roll[i]) and ema50_1w_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below Donchian mid OR volume filter fails
            if close[i] < donchian_mid[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Donchian mid OR volume filter fails
            if close[i] > donchian_mid[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals