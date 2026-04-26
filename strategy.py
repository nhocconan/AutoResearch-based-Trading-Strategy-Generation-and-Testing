#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeConfirmation
Hypothesis: 4h breakout above/below 20-period Donchian channels in direction of 12h EMA50 trend, confirmed by volume spike (>1.5x 20-bar MA). Uses Donchian structure (proven edge from DB) with 12h trend filter to avoid counter-trend trades. Volume confirmation reduces false breakouts. Designed for 19-50 trades/year (75-200 total over 4 years) to avoid fee drag. Works in both bull and bear markets by following the 12h trend while using Donchian breakouts for precise entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian channels (20-period) on 4h
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations (20 for Donchian, 20 for volume, 50 for EMA)
    start_idx = max(20, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(high_ma[i]) or 
            np.isnan(low_ma[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            bars_since_entry += 1 if position != 0 else 0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_12h_aligned[i]
        upper_channel = high_ma[i]
        lower_channel = low_ma[i]
        vol_spike = volume_spike[i]
        
        # Determine 12h trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_12h = close_val > ema_50_val
        bearish_12h = close_val < ema_50_val
        
        # Entry conditions: breakout of Donchian channel in trend direction with volume
        long_entry = (close_val > upper_channel) and bullish_12h and vol_spike
        short_entry = (close_val < lower_channel) and bearish_12h and vol_spike
        
        # Exit conditions: opposite Donchian channel touch (or trend reversal)
        exit_long = (close_val < lower_channel) or not bullish_12h
        exit_short = (close_val > upper_channel) or not bearish_12h
        
        # Minimum holding period: 2 bars
        min_hold = 2
        
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

name = "4h_Donchian20_Breakout_12hTrend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0