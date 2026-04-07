#!/usr/bin/env python3
"""
1D Donchian Breakout with Volume Confirmation and Weekly Trend Filter
Long when price breaks above weekly Donchian upper band with volume confirmation and weekly trend up
Short when price breaks below weekly Donchian lower band with volume confirmation and weekly trend down
Exit when price crosses back to weekly EMA(50)
Uses weekly Donchian channels (20-period) as trend filter and structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_volume_1w_trend_v1"
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
    
    # === Weekly Donchian Channel (20-period high/low) ===
    df_1w = get_htf_data(prices, '1w')
    donch_high = df_1w['high'].rolling(window=20, min_periods=20).max().values
    donch_low = df_1w['low'].rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    # === Weekly EMA(50) for exit ===
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # === Volume confirmation (daily) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)  # Avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses back below weekly EMA(50)
            if close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back above weekly EMA(50)
            if close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with volume confirmation
            if close[i] > donch_high_aligned[i] and volume[i] > vol_ma[i]:
                # Breakout above weekly Donchian high with volume -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donch_low_aligned[i] and volume[i] > vol_ma[i]:
                # Breakdown below weekly Donchian low with volume -> short
                position = -1
                signals[i] = -0.25
    
    return signals