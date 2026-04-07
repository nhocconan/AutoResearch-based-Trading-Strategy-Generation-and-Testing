# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
12h Donchian Breakout with Daily Trend Filter and Volume Confirmation
Long when price breaks above 20-period Donchian upper with volume > 1.5x average and daily EMA up
Short when price breaks below 20-period Donchian lower with volume > 1.5x average and daily EMA down
Exit when price crosses back to 20-period EMA midline
Designed for low trade frequency (target: 50-150 trades over 4 years) to minimize fee drag.
Works in both bull and bear markets via trend filter and volatility-adaptive breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_daily_trend_volume_v1"
timeframe = "12h"
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
    
    # === Donchian Channels (20-period high/low) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Exit: 20-period EMA midline ===
    ema_mid = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)  # Avoid division by zero
    
    # === Daily trend filter (EMA 21) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_mid[i]) or np.isnan(vol_ratio[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses back below EMA midline
            if close[i] < ema_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back above EMA midline
            if close[i] > ema_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need expanding volume (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with volume confirmation AND daily trend filter
            if close[i] > donchian_high[i] and ema_1d_aligned[i] > ema_1d_aligned[i-1]:
                # Breakout above upper channel with rising daily EMA -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donchian_low[i] and ema_1d_aligned[i] < ema_1d_aligned[i-1]:
                # Breakdown below lower channel with falling daily EMA -> short
                position = -1
                signals[i] = -0.25
    
    return signals