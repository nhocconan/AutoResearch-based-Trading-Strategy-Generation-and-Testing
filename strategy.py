#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1wTrend_1dVolSpike
Hypothesis: 6h Donchian(20) breakout with 1w trend filter (price > 1w EMA50) and 1d volume spike (>2.0x 20-bar MA). Uses weekly structure for major trend alignment to avoid counter-trend trades in bear markets, Donchian for objective breakout levels, and daily volume confirmation to filter false breakouts. Designed for 12-30 trades/year (50-120 total over 4 years) to minimize fee drag. Works in bull/bear markets by following 1w trend while using Donchian breakouts for entries and volume spike for confirmation.
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
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume MA for spike detection
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Donchian channels (20-period) on 6h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    base_size = 0.25  # Position size (25% of capital)
    
    # Warmup: max of calculations (20 for Donchian, 50 for 1w EMA, 20 for 1d vol MA)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            bars_since_entry += 1 if position != 0 else 0
            continue
        
        close_val = close[i]
        highest_high_val = highest_high[i]
        lowest_low_val = lowest_low[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_ma_val = vol_ma_1d_aligned[i]
        vol_spike = volume[i] > (vol_ma_val * 2.0)
        
        # Determine 1w trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_1w = close_val > ema_50_val
        bearish_1w = close_val < ema_50_val
        
        # Entry conditions: Donchian breakout in 1w trend direction with volume spike
        long_entry = (close_val > highest_high_val) and bullish_1w and vol_spike
        short_entry = (close_val < lowest_low_val) and bearish_1w and vol_spike
        
        # Exit conditions: opposite Donchian level touch (opposite channel)
        exit_long = close_val < lowest_low_val
        exit_short = close_val > highest_high_val
        
        # Minimum holding period: 3 bars (to avoid whipsaw on breakouts)
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

name = "6h_Donchian20_Breakout_1wTrend_1dVolSpike"
timeframe = "6h"
leverage = 1.0