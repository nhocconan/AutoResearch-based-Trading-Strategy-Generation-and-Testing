#!/usr/bin/env python3
# 1d_1w_donchian_breakout_volume_v1
# Hypothesis: Use weekly Donchian channels (20-period) for trend direction and daily Donchian breakouts for entries, with volume confirmation. Long when price breaks above weekly upper band and daily upper band with volume > 1.5x average; short when price breaks below weekly lower band and daily lower band with volume confirmation. This captures strong momentum moves while filtering false breakouts. Works in both bull and bear markets by following the weekly trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (Donchian channels)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high_20_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_20_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high_20_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20_1w)
    donchian_low_20_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20_1w)
    
    # Calculate daily Donchian breakout levels (20-period)
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(donchian_high_20_1w_aligned[i]) or np.isnan(donchian_low_20_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly lower band or daily lower band
            if close[i] < donchian_low_20_1w_aligned[i] or close[i] < donchian_low_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly upper band or daily upper band
            if close[i] > donchian_high_20_1w_aligned[i] or close[i] > donchian_high_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above weekly upper band AND daily upper band with volume
            if (close[i] > donchian_high_20_1w_aligned[i] and 
                close[i] > donchian_high_20[i] and 
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below weekly lower band AND daily lower band with volume
            elif (close[i] < donchian_low_20_1w_aligned[i] and 
                  close[i] < donchian_low_20[i] and 
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals