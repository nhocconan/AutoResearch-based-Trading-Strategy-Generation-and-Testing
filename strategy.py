#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout (20-period) with 1d weekly pivot (W1) direction filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND 1d price > 1w pivot point (bullish weekly bias) AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian(20) low AND 1d price < 1w pivot point (bearish weekly bias) AND volume > 1.5x 20-period average.
# Exit when price crosses back through Donchian midpoint or volume drops below average.
# Weekly pivot from 1w data provides structural bias to avoid counter-trend trades in strong trends.
# Volume filter ensures participation and avoids low-conviction moves.
# Designed for 6h timeframe with target: 15-30 trades/year to avoid fee drag.
name = "6h_Donchian_W1Pivot_VolumeFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_roll + low_roll) / 2.0
    
    # Weekly pivot point calculation from 1w data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly pivot: (H + L + C) / 3 using weekly close
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    pivot_point = (weekly_high + weekly_low + weekly_close) / 3.0
    pivot_point_aligned = align_htf_to_ltf(prices, df_1w, pivot_point)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Sufficient warmup for Donchian
    
    for i in range(start_idx, n):
        if np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(donchian_mid[i]) or \
           np.isnan(pivot_point_aligned[i]) or np.isnan(volume_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Donchian high, price > weekly pivot (bullish bias), volume filter
            long_cond = (close[i] > high_roll[i]) and (close[i] > pivot_point_aligned[i]) and volume_filter[i]
            # Short conditions: break below Donchian low, price < weekly pivot (bearish bias), volume filter
            short_cond = (close[i] < low_roll[i]) and (close[i] < pivot_point_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below Donchian midpoint or volume filter fails
            if close[i] < donchian_mid[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Donchian midpoint or volume filter fails
            if close[i] > donchian_mid[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals