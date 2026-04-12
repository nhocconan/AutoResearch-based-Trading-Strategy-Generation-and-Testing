#!/usr/bin/env python3
"""
4h_1d_Multi_Signal_Confluence_v1
Hypothesis: Combine Donchian(20) breakout, 1d EMA200 trend, and volume spike for high-probability entries.
Long when price breaks above Donchian high AND above 1d EMA200 AND volume > 1.5x 20-period average.
Short when price breaks below Donchian low AND below 1d EMA200 AND volume > 1.5x 20-period average.
Exit when price crosses Donchian midpoint or opposite breakout occurs.
Uses tight entry conditions to limit trades (~30-50/year) and reduce fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Multi_Signal_Confluence_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 1d EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    ema200 = pd.Series(daily_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200)
    
    # Volume spike filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema200_aligned[i]) or np.isnan(vol_ma20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions
        long_entry = (close[i] > donchian_high[i] and 
                     close[i] > ema200_aligned[i] and 
                     volume_spike[i])
        short_entry = (close[i] < donchian_low[i] and 
                      close[i] < ema200_aligned[i] and 
                      volume_spike[i])
        
        # Exit conditions
        long_exit = close[i] < donchian_mid[i]
        short_exit = close[i] > donchian_mid[i]
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals