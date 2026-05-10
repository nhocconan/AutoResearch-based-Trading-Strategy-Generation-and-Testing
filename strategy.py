#!/usr/bin/env python3
# 6h_Donchian_Breakout_1dTrend_Volume
# Hypothesis: 6-hour Donchian(20) breakout with 1-day EMA trend filter and volume confirmation.
# Works in bull/bear by requiring trend alignment, reducing false breakouts.
# Uses discrete sizing (0.25) to limit turnover. Targets 50-150 trades over 4 years.
# Breakouts occur when price closes above/below the 20-period high/low with volume surge.

name = "6h_Donchian_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 6h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 20-period Donchian channels
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume average (10-period = ~10 periods of 6h bars)
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    # Warmup: need Donchian (20) + EMA34 (34) + volume MA (10)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(high_max[i]) or
            np.isnan(low_min[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        # Determine trend from 1d EMA34
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
        uptrend = close_1d_aligned[i] > ema_34_1d_aligned[i]
        downtrend = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation (1.5x average)
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        # Donchian breakout/breakdown
        breakout_high = close[i] > high_max[i]
        breakdown_low = close[i] < low_min[i]
        
        if position == 0:
            bars_since_entry = 0
            # Long: Breakout above Donchian high with volume surge and 1d uptrend
            if breakout_high and volume_surge and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below Donchian low with volume surge and 1d downtrend
            elif breakdown_low and volume_surge and downtrend:
                signals[i] = -0.25
                position = -1
        else:
            bars_since_entry += 1
            # Enforce minimum holding period of 2 bars (12 hours)
            if bars_since_entry < 2:
                signals[i] = signals[i-1]  # maintain position
                continue
            
            if position == 1:
                # Long exit: price breaks below Donchian low or trend changes
                if close[i] < low_min[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price breaks above Donchian high or trend changes
                if close[i] > high_max[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals