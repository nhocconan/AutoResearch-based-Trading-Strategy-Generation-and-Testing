#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeConfirmation
Hypothesis: 4h breakout above/below 20-period Donchian channel in direction of 1d EMA50 trend, confirmed by volume spike (>1.5x 20-bar MA). Uses Donchian breakouts (proven edge) with 1d trend filter to avoid counter-trend trades. Volume confirmation reduces false breakouts. Designed for 20-50 trades/year (75-200 total over 4 years) to avoid fee drag. Works in both bull and bear markets by following the 1d trend while using Donchian structure for precise entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channel (20-period) on 4h
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations (20 for Donchian, 20 for vol, 50 for ema)
    start_idx = max(20, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            bars_since_entry += 1 if position != 0 else 0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_1d_aligned[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        vol_spike = volume_spike[i]
        
        # Determine 1d trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_1d = close_val > ema_50_val
        bearish_1d = close_val < ema_50_val
        
        # Entry conditions: breakout of Donchian channel in trend direction with volume
        # Upper break = long, lower break = short
        long_entry = (close_val > upper) and bullish_1d and vol_spike
        short_entry = (close_val < lower) and bearish_1d and vol_spike
        
        # Exit conditions: opposite Donchian touch (or trend reversal)
        exit_long = (close_val < lower) or not bullish_1d
        exit_short = (close_val > upper) or not bearish_1d
        
        # Minimum holding period: 3 bars
        min_hold = 3
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
                bars_since_entry = 0
            elif short_entry:
                signals[i] = -base_size
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
        elif position == 1:
            # Long - check exit conditions only after minimum hold
            if bars_since_entry >= min_hold and exit_long:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = base_size
                bars_since_entry += 1
        elif position == -1:
            # Short - check exit conditions only after minimum hold
            if bars_since_entry >= min_hold and exit_short:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -base_size
                bars_since_entry += 1
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0