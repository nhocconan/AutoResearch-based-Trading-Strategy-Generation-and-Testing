#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Confirmation and 12h Trend Filter
Long when price breaks above Donchian upper with volume spike and 12h trend up
Short when price breaks below Donchian lower with volume spike and 12h trend down
Exit on opposite breakout or volume drop
Works in both bull and bear markets by following the 12h trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_12h_trend_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Donchian Channel (20-period) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === 12h Trend Filter (EMA 50) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_ratio[i]) or np.isnan(ema_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian lower
            if close[i] < donch_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper
            if close[i] > donch_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume spike
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with 12h trend confirmation
            if close[i] > donch_high[i] and ema_12h_aligned[i] > ema_12h_aligned[i-1]:
                # Breakout above upper band with rising 12h trend -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donch_low[i] and ema_12h_aligned[i] < ema_12h_aligned[i-1]:
                # Breakdown below lower band with falling 12h trend -> short
                position = -1
                signals[i] = -0.25
    
    return signals