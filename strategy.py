#!/usr/bin/env python3
"""
4H Donchian Breakout with Volume Confirmation and 12h Trend Filter
Long when price breaks above Donchian upper band (20-period high) with expanding volume AND 12h EMA trend up
Short when price breaks below Donchian lower band (20-period low) with expanding volume AND 12h EMA trend down
Exit when price crosses back to midline (10-period average of high/low)
Uses Donchian channels that adapt to volatility, reducing false breakouts in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_12h_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Donchian Channels (20-period high/low) ===
    # Upper band: 20-period high
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Middle line: average of upper and lower
    donch_mid = (donch_upper + donch_lower) / 2
    
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
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or 
            np.isnan(donch_mid[i]) or np.isnan(vol_ratio[i]) or np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses back below middle line
            if close[i] < donch_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price crosses back above middle line
            if close[i] > donch_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Need expanding volume (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with volume confirmation AND 12h trend filter
            if close[i] > donch_upper[i] and ema_12h_aligned[i] > ema_12h_aligned[i-1]:
                # Breakout above upper channel with rising 12h EMA -> long
                position = 1
                signals[i] = 0.30
            elif close[i] < donch_lower[i] and ema_12h_aligned[i] < ema_12h_aligned[i-1]:
                # Breakdown below lower channel with falling 12h EMA -> short
                position = -1
                signals[i] = -0.30
    
    return signals