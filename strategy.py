#!/usr/bin/env python3
"""
12h Donchian Breakout with Volume Confirmation and 1d Trend Filter
Long when price breaks above Donchian(20) high with above-average volume and 1d EMA50 uptrend
Short when price breaks below Donchian(20) low with above-average volume and 1d EMA50 downtrend
Exit when price crosses Donchian midpoint or trend changes
Designed for 12h timeframe to capture multi-day trends while avoiding whipsaws
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_volume_1d_trend_v1"
timeframe = "12h"
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
    
    # === 1d Trend Filter (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Donchian Channels (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # === Volume Confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian midpoint OR 1d trend turns down
            if close[i] < donchian_mid[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian midpoint OR 1d trend turns up
            if close[i] > donchian_mid[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need expanding volume (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with volume confirmation and 1d trend alignment
            if close[i] > donchian_high[i] and close[i] > ema_50_1d_aligned[i]:
                # Break above upper band with uptrend -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donchian_low[i] and close[i] < ema_50_1d_aligned[i]:
                # Break below lower band with downtrend -> short
                position = -1
                signals[i] = -0.25
    
    return signals