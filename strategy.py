#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Confirmation and 12h Trend Filter
Long when price breaks above Donchian(20) high with volume > 1.5x 20-period average AND 12h EMA(21) trending up
Short when price breaks below Donchian(20) low with volume > 1.5x 20-period average AND 12h EMA(21) trending down
Exit when price crosses back to Donchian midline (10-period average)
Designed to work in both bull and bear markets by requiring trend alignment and volume confirmation.
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
    
    # === Donchian Channels (20-period) ===
    # Upper band: 20-period high
    high_series = pd.Series(high)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    low_series = pd.Series(low)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    # Middle line: average of upper and lower
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
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
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(vol_ratio[i]) or np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses back below middle line
            if close[i] < donchian_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back above middle line
            if close[i] > donchian_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with volume confirmation AND 12h trend filter
            if close[i] > donchian_upper[i] and ema_12h_aligned[i] > ema_12h_aligned[i-1]:
                # Breakout above upper channel with rising 12h EMA -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donchian_lower[i] and ema_12h_aligned[i] < ema_12h_aligned[i-1]:
                # Breakdown below lower channel with falling 12h EMA -> short
                position = -1
                signals[i] = -0.25
    
    return signals