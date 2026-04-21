#!/usr/bin/env python3
"""
4h_1d_Donchian20_Breakout_Volume_Trend_HTF
Hypothesis: 4-hour Donchian(20) breakouts with 1-day EMA(50) trend filter and volume confirmation work in both bull and bear markets by capturing strong momentum moves aligned with higher timeframe trend. Target: 20-50 trades/year for low fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1-day data once for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on daily close
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Main timeframe data (4h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Volume filter: current volume > 1.8x 30-period average
    volume_avg = np.full(n, np.nan)
    for i in range(30, n):
        volume_avg[i] = np.mean(volume[i-30:i])
    volume_filter = volume > (1.8 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if NaN in critical values
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        trend = ema_50_1d_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper with volume and above daily EMA50
            if price > upper and vol_ok and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower with volume and below daily EMA50
            elif price < lower and vol_ok and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to Donchian lower or breaks below EMA50
            if price < lower or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to Donchian upper or breaks above EMA50
            if price > upper or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Donchian20_Breakout_Volume_Trend_HTF"
timeframe = "4h"
leverage = 1.0