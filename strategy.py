#!/usr/bin/env python3
"""
4H Donchian Breakout with Volume Confirmation and 1D Trend Filter
Long when price breaks above Donchian upper band (20-period high) with volume > 1.5x average AND 1D EMA trend up
Short when price breaks below Donchian lower band (20-period low) with volume > 1.5x average AND 1D EMA trend down
Exit when price crosses back to midline (10-period EMA)
Uses Donchian channels for breakout detection and volume confirmation to filter false breakouts.
Designed for 4h timeframe with 1d trend filter to capture medium-term trends while avoiding counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_1d_trend_v1"
timeframe = "4h"
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
    
    # === Donchian Channels (20-period) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Exit: 10-period EMA midline ===
    ema_mid = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === 1D trend filter (EMA 21) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_mid[i]) or np.isnan(vol_ratio[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses back below midline
            if close[i] < ema_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back above midline
            if close[i] > ema_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need above-average volume (at least 1.5x)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with volume confirmation AND 1D trend filter
            if close[i] > donch_high[i] and ema_1d_aligned[i] > ema_1d_aligned[i-1]:
                # Breakout above upper band with rising 1D EMA -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donch_low[i] and ema_1d_aligned[i] < ema_1d_aligned[i-1]:
                # Breakdown below lower band with falling 1D EMA -> short
                position = -1
                signals[i] = -0.25
    
    return signals