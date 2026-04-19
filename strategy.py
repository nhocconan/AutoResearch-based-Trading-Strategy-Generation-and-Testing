#!/usr/bin/env python3
"""
1h_4h_Donchian_Breakout_1d_Volume_Spike
Hypothesis: 1h timeframe uses 4h Donchian channel breakout for direction with 1h volume spike confirmation.
4h trend provides higher probability moves, 1h volume spike confirms institutional participation.
Works in bull/bear by requiring volume confirmation on breakouts, avoiding false signals in chop.
Target: 15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drag.
"""

name = "1h_4h_Donchian_Breakout_1d_Volume_Spike"
timeframe = "1h"
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
    
    # 4h Donchian channel (20-period) for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian on 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align to 1h timeframe
    donch_high_1h = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_1h = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # 1h volume spike confirmation (volume > 2.0 * 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    # Session filter: 08:00-20:00 UTC only
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need enough data for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(donch_high_1h[i]) or np.isnan(donch_low_1h[i]) or 
            np.isnan(volume_ma[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian high with volume spike
            if (close[i] > donch_high_1h[i] and volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian low with volume spike
            elif (close[i] < donch_low_1h[i] and volume_spike[i]):
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below 4h Donchian low
            if close[i] < donch_low_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if price breaks above 4h Donchian high
            if close[i] > donch_high_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals