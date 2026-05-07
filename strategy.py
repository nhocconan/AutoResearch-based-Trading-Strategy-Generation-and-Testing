#!/usr/bin/env python3
"""
4H_Donchian_20_VolumeTrend_12H_EMA50
Hypothesis: 4h Donchian breakout (20-period) with 12h EMA50 trend filter and volume confirmation works in both bull and bear markets by capturing strong directional moves while avoiding whipsaws. Volume filter ensures breakouts have conviction, EMA50 filters counter-trend noise. Targets 20-50 trades/year on 4h timeframe to minimize fee drag.
"""
name = "4H_Donchian_20_VolumeTrend_12H_EMA50"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend direction
    close_12h_series = pd.Series(df_12h['close'])
    ema_50 = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current 4h volume > 1.5 x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(50, 20)  # Ensure sufficient warmup for 12h EMA50 and Donchian
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 48 bars between trades (8 days on 4h TF) to reduce frequency
            if bars_since_exit < 48:
                continue
                
            # Long: price breaks above Donchian high with 12h EMA50 uptrend and volume spike
            if (close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1] and 
                close[i] > ema_50_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price breaks below Donchian low with 12h EMA50 downtrend and volume spike
            elif (close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1] and 
                  close[i] < ema_50_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite Donchian level (mean reversion within channel)
            if position == 1 and close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals