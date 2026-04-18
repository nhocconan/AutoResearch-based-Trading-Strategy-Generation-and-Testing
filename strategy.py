#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_Volume_Trend
Hypothesis: Donchian channel breakout from 12h price action with volume confirmation and 1d trend filter.
Long when price breaks above 20-period upper band with volume spike and 1d EMA50 uptrend.
Short when price breaks below 20-period lower band with volume spike and 1d EMA50 downtrend.
Uses tight entry conditions to target 12-37 trades/year, avoiding overtrading while capturing
strong momentum moves in both bull and bear markets via trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels (20-period) on 12h data
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: >2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Warmup for Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = high_max[i]
        lower = low_min[i]
        ema50 = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: break above upper band with volume spike and 1d uptrend
            if (price > upper and
                vol_spike and
                price > ema50):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with volume spike and 1d downtrend
            elif (price < lower and
                  vol_spike and
                  price < ema50):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price re-enters Donchian channel or trend reverses
            if price < upper and price > lower or price < ema50:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price re-enters Donchian channel or trend reverses
            if price < upper and price > lower or price > ema50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0