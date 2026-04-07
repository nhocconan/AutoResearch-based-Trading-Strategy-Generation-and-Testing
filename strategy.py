#!/usr/bin/env python3
"""
4H Donchian Breakout with Volume Confirmation and 12h Trend Filter
Long when price breaks above Donchian high (20) with expanding volume AND 12h EMA trend up
Short when price breaks below Donchian low (20) with expanding volume AND 12h EMA trend down
Exit when price crosses back to Donchian mid (10)
Uses price channel breakouts that capture strong moves while reducing false signals in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_12h_trend_v1"
timeframe = "4h"
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
    
    # === Donchian Channels (20-period high/low) ===
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)  # Avoid division by zero
    
    # === 12h trend filter (EMA 21) ===
    df_12h = get_htf_data(prices, '12h')
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(vol_ratio[i]) or np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses back below Donchian mid
            if close[i] < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price crosses back above Donchian mid
            if close[i] > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Need expanding volume (above average)
            if vol_ratio[i] < 1.3:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with volume confirmation AND 12h trend filter
            if close[i] > high_20[i] and ema_12h_aligned[i] > ema_12h_aligned[i-1]:
                # Breakout above Donchian high with rising 12h EMA -> long
                position = 1
                signals[i] = 0.30
            elif close[i] < low_20[i] and ema_12h_aligned[i] < ema_12h_aligned[i-1]:
                # Breakdown below Donchian low with falling 12h EMA -> short
                position = -1
                signals[i] = -0.30
    
    return signals