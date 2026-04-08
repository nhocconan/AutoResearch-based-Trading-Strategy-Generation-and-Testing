#!/usr/bin/env python3
# 4h_1d_donchian_breakout_volume_v3
# Hypothesis: Use 1d Donchian channels for long-term trend direction, 4h Donchian breakout for entry, and volume confirmation for institutional participation.
# Works in bull markets (trend continuation) and bear markets (mean reversion from oversold/overbought levels).
# Target: 25-40 trades/year per symbol (100-160 total over 4 years) by requiring multi-timeframe alignment and volume filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (entry signals)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for long-term trend context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period) for entry/exit
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate 1d Donchian channels (55-period) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high_1d = pd.Series(high_1d).rolling(window=55, min_periods=55).max().values
    donchian_low_1d = pd.Series(low_1d).rolling(window=55, min_periods=55).min().values
    donchian_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    
    # Volume confirmation: volume > 1.8x average of last 48 periods (2 days in 4h)
    vol_ma = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    vol_confirm = volume > vol_ma * 1.8
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(donchian_high_1d_aligned[i]) or np.isnan(donchian_low_1d_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 4h Donchian low or breaks 1d lower band
            if close[i] < donchian_low_aligned[i] or close[i] < donchian_low_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h Donchian high or breaks 1d upper band
            if close[i] > donchian_high_aligned[i] or close[i] > donchian_high_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above 4h Donchian high with 1d uptrend bias and volume
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > donchian_low_1d_aligned[i] and  # Above 1d lower band (not in extreme oversold)
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 4h Donchian low with 1d downtrend bias and volume
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < donchian_high_1d_aligned[i] and  # Below 1d upper band (not in extreme overbought)
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals