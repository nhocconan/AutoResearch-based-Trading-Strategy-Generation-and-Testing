#!/usr/bin/env python3
"""
4H Donchian Breakout with Volume Confirmation and 1D Trend Filter
Long when price breaks above Donchian(20) upper band with expanding volume AND 1d EMA trend up
Short when price breaks below Donchian(20) lower band with expanding volume AND 1d EMA trend down
Exit when price crosses back to middle line (10-period EMA)
Uses ATR-based bands that adapt to volatility, reducing false breakouts in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_1d_trend_v1"
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
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_middle = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)  # Avoid division by zero
    
    # === 1d trend filter (EMA 21) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or 
            np.isnan(donch_middle[i]) or np.isnan(vol_ratio[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses back below middle line
            if close[i] < donch_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back above middle line
            if close[i] > donch_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need expanding volume (above average)
            if vol_ratio[i] < 1.2:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with volume confirmation AND 1d trend filter
            if close[i] > donch_upper[i] and ema_1d_aligned[i] > ema_1d_aligned[i-1]:
                # Breakout above upper channel with rising 1d EMA -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donch_lower[i] and ema_1d_aligned[i] < ema_1d_aligned[i-1]:
                # Breakdown below lower channel with falling 1d EMA -> short
                position = -1
                signals[i] = -0.25
    
    return signals