#!/usr/bin/env python3
"""
4h Donchian Breakout + Volume + 1D Trend Filter
Hypothesis: Donchian breakouts with volume confirmation in the direction of the 1D trend capture strong momentum moves while avoiding counter-trend trades. This works in both bull and bear markets because the 1D trend filter ensures we only trade with the dominant higher timeframe momentum, reducing whipsaws. Volume confirmation ensures breakouts have institutional participation. Target: 20-30 trades/year to minimize fee drag.
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
    
    # Get 1D data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1D EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.8x 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        trend = ema50_1d_aligned[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume, in uptrend
            if price > donchian_high[i] and vol_ok and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume, in downtrend
            elif price < donchian_low[i] and vol_ok and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns to Donchian midpoint or trend weakens
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if price < midpoint or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to Donchian midpoint or trend weakens
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if price > midpoint or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_1DTrend"
timeframe = "4h"
leverage = 1.0