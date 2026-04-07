#!/usr/bin/env python3
"""
4H Donchian Breakout with Volume Confirmation and Daily Trend Filter
Long when price breaks above Donchian upper (20-bar high) with expanding volume AND daily EMA trend up
Short when price breaks below Donchian lower (20-bar low) with expanding volume AND daily EMA trend down
Exit when price crosses back to midline (10-bar average of high/low)
Designed for fewer trades (<50/year) to minimize fee drag while capturing strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_1d_trend_v2"
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
    # Upper: 20-bar high, Lower: 20-bar low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    # Midline: average of upper and lower
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # === Volume confirmation (20-bar average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === Daily trend filter (EMA 21) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(vol_ratio[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses back below midline
            if close[i] < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back above midline
            if close[i] > donchian_mid[i]:
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
            if close[i] > donchian_upper[i] and ema_1d_aligned[i] > ema_1d_aligned[i-1]:
                # Breakout above upper channel with rising daily EMA -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donchian_lower[i] and ema_1d_aligned[i] < ema_1d_aligned[i-1]:
                # Breakdown below lower channel with falling daily EMA -> short
                position = -1
                signals[i] = -0.25
    
    return signals