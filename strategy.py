#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian breakout with 1-day volume confirmation and 1-week trend filter.
Long when price breaks above Donchian(20) high, 1-day volume > 1.5x average, and 1-week EMA200 rising.
Short when price breaks below Donchian(20) low, 1-day volume > 1.5x average, and 1-week EMA200 falling.
Exit when price returns to Donchian midpoint or 1-week EMA200 reverses.
Designed for low trade frequency (~20-40/year) to capture strong trends while minimizing whipsaws.
Works in both bull and bear markets by requiring strong trend confirmation (1-week EMA200).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for volume confirmation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load 1-week data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1-day average volume (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1-week EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align HTF indicators to lower timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate Donchian channels (20-period) on price data
    # Upper band: highest high of last 20 periods
    high_series = pd.Series(high)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    
    # Lower band: lowest low of last 20 periods
    low_series = pd.Series(low)
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Middle band: average of upper and lower
    donchian_mid = (donchian_high + donchian_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after warmup period (EMA200 needs 200 periods)
        # Skip if data not ready
        if (np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ma_val = vol_ma_1d_aligned[i]
        ema200_val = ema200_1w_aligned[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high, volume confirmation, EMA200 rising
            if (close[i] > donchian_high[i] and 
                vol_current > 1.5 * vol_ma_val and
                ema200_val > ema200_1w_aligned[i-1]):  # EMA200 rising
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low, volume confirmation, EMA200 falling
            elif (close[i] < donchian_low[i] and 
                  vol_current > 1.5 * vol_ma_val and
                  ema200_val < ema200_1w_aligned[i-1]):  # EMA200 falling
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to Donchian midpoint OR EMA200 starts falling
                if (close[i] < donchian_mid[i] or ema200_val < ema200_1w_aligned[i-1]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to Donchian midpoint OR EMA200 starts rising
                if (close[i] > donchian_mid[i] or ema200_val > ema200_1w_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dVolume_1wEMA200_Trend"
timeframe = "4h"
leverage = 1.0