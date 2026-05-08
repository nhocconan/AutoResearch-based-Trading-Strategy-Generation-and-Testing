#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above 1d upper Donchian band AND 1w EMA50 rising AND volume > 2x 20-period average.
# Short when price breaks below 1d lower Donchian band AND 1w EMA50 falling AND volume > 2x 20-period average.
# Exit when price crosses back inside Donchian bands (lower < price < upper).
# This strategy captures multi-day breakouts aligned with weekly trend, reducing false signals.
# Weekly EMA50 filter ensures we trade with the longer-term trend. Volume confirms institutional participation.
# Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the weekly trend direction.

name = "1d_Donchian_20_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily Donchian(20) - using previous day's data to avoid look-ahead
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1w EMA50 direction
    ema50_rising = np.zeros_like(ema50_1w_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_1w_aligned, dtype=bool)
    ema50_rising[1:] = ema50_1w_aligned[1:] > ema50_1w_aligned[:-1]
    ema50_falling[1:] = ema50_1w_aligned[1:] < ema50_1w_aligned[:-1]
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Sufficient warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(ema50_rising[i]) or 
            np.isnan(ema50_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian, 1w EMA50 rising, volume filter
            long_cond = (close[i] > high_20[i]) and ema50_rising[i] and volume_filter[i]
            # Short conditions: price breaks below lower Donchian, 1w EMA50 falling, volume filter
            short_cond = (close[i] < low_20[i]) and ema50_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below lower Donchian band
            if close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above upper Donchian band
            if close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals