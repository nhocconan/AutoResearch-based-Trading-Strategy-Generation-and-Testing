#!/usr/bin/env python3
"""
12h_Donchian_Breakout_Volume_TrendFilter
Hypothesis: Price breaks above/below 20-period Donchian channel on 12h timeframe with volume confirmation and 1-day EMA trend filter.
Captures breakouts in trending markets while using volume and higher timeframe trend to filter false signals.
Target: 15-25 trades/year to minimize fee drag while capturing strong directional moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian channel (20-period)
    donchian_window = 20
    high_max = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    low_min = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 1-day EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(donchian_window, 20, 50)  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = high_max[i]
        lower = low_min[i]
        ema50 = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: break above upper band with volume and uptrend
            if price > upper and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with volume and downtrend
            elif price < lower and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price closes below lower band OR trend turns down
            if price < lower:
                signals[i] = 0.0
                position = 0
            elif price < ema50:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price closes above upper band OR trend turns up
            if price > upper:
                signals[i] = 0.0
                position = 0
            elif price > ema50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian_Breakout_Volume_TrendFilter"
timeframe = "12h"
leverage = 1.0