#!/usr/bin/env python3
# 1d_1w_donchian_breakout_volume_v1
# Hypothesis: Use 1w Donchian channels for long-term trend direction and 1d Donchian breakout for entry, with volume confirmation to filter institutional participation. Works in bull markets (trend continuation) and bear markets (mean reversion from oversold/overbought levels). Target: 10-25 trades/year per symbol (40-100 total over 4 years) by requiring multi-timeframe alignment and volume filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels (entry signals)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for long-term trend context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period) for entry/exit
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 1w Donchian channels (20-period) for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    
    # Volume confirmation: volume > 1.5x average of last 20 periods (20 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(donchian_high_1w_aligned[i]) or np.isnan(donchian_low_1w_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 1d Donchian low or breaks 1w lower band
            if close[i] < donchian_low_aligned[i] or close[i] < donchian_low_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above 1d Donchian high or breaks 1w upper band
            if close[i] > donchian_high_aligned[i] or close[i] > donchian_high_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above 1d Donchian high with 1w uptrend bias and volume
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > donchian_low_1w_aligned[i] and  # Above 1w lower band (not in extreme oversold)
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 1d Donchian low with 1w downtrend bias and volume
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < donchian_high_1w_aligned[i] and  # Below 1w upper band (not in extreme overbought)
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals