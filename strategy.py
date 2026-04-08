#!/usr/bin/env python3
"""
6h_1d_price_channel_volume_v3
Hypothesis: Use 1-day price channels (Donchian) to capture trend with volume confirmation.
Long when price breaks above 1-day high with volume > 1.5x avg. Short when breaks below 1-day low with volume > 1.5x avg.
Exit when price returns to middle of channel (mean reversion within day).
Designed to work in trending markets with volume confirmation to avoid false breakouts.
Target: 15-35 trades/year per symbol (60-140 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_price_channel_volume_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for price channel calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-day lookback)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 20-period high and low (using previous 20 days, not including current)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Middle of channel
    mid_20 = (high_20 + low_20) / 2
    
    # Align daily channels to 6h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    mid_20_aligned = align_htf_to_ltf(prices, df_1d, mid_20)
    
    # Volume confirmation: volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(mid_20_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to middle of channel or breaks below low
            if close[i] <= mid_20_aligned[i] or close[i] < low_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price returns to middle of channel or breaks above high
            if close[i] >= mid_20_aligned[i] or close[i] > high_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above 20-day high with volume confirmation
            if close[i] > high_20_aligned[i] and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 20-day low with volume confirmation
            elif close[i] < low_20_aligned[i] and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals