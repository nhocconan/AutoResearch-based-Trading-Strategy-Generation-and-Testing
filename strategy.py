#!/usr/bin/env python3
# 6h_donchian_breakout_1d_trend_volume_v1
# Hypothesis: 6h timeframe trading using Donchian channel breakouts with 1d trend filter and volume confirmation.
# Donchian breakouts capture momentum in both trending and ranging markets. The 1d EMA50 filter ensures
# alignment with daily trend to avoid counter-trend trades. Volume confirmation (>2x 20-period average)
# ensures breakouts are backed by participation. Target: 12-37 trades/year per symbol.

name = "6h_donchian_breakout_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter - call ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian channels (20-period) on 6h data
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned daily EMA50 for current 6h bar
        ema50_val = ema50_1d_aligned[i]
        
        # Skip if any required data is NaN
        if np.isnan(ema50_val) or np.isnan(high_roll_max[i]) or np.isnan(low_roll_min[i]) or np.isnan(vol_ma[i]) or volume[i] == 0:
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = high[i] >= high_roll_max[i]
        breakout_down = low[i] <= low_roll_min[i]
        
        # Volume breakout condition: current volume > 2.0x 20-period average
        vol_breakout = volume[i] > 2.0 * vol_ma[i]
        
        # Trend filter: price above/below daily EMA50
        uptrend = close[i] > ema50_val
        downtrend = close[i] < ema50_val
        
        if position == 1:  # Long position
            # Exit if price breaks below Donchian lower band
            if low[i] <= low_roll_min[i]:
                position = 0
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian upper band
            if high[i] >= high_roll_max[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long above Donchian upper band with volume confirmation and uptrend
            if breakout_up and vol_breakout and uptrend:
                position = 1
                signals[i] = 0.25
            # Breakout short below Donchian lower band with volume confirmation and downtrend
            elif breakout_down and vol_breakout and downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals