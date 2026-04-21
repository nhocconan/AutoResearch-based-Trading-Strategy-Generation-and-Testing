#!/usr/bin/env python3
"""
1d_Donchian_Breakout_Volume_Spike
Hypothesis: Donchian breakout on 1d timeframe with volume spike confirmation.
Trades only when price breaks above the 20-day high with volume > 1.5x 20-day average,
or breaks below the 20-day low with volume > 1.5x 20-day average.
Designed for low trade frequency (target 10-25 trades/year) to minimize fee drag
while capturing major trends in both bull and bear markets. Volume spike filters
out false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d OHLCV data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d Donchian channels (20) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Donchian high and low
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Average volume
    avg_volume = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to lower timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    avg_volume_aligned = align_htf_to_ltf(prices, df_1d, avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(avg_volume_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        volume_current = prices['volume'].iloc[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        avg_volume_val = avg_volume_aligned[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume spike
            if (price_close > donchian_high_val and 
                volume_current > 1.5 * avg_volume_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with volume spike
            elif (price_close < donchian_low_val and 
                  volume_current > 1.5 * avg_volume_val):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price crosses the opposite Donchian band
            if position == 1 and price_close < donchian_low_val:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > donchian_high_val:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian_Breakout_Volume_Spike"
timeframe = "1d"
leverage = 1.0