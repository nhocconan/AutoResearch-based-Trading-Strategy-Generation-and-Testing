#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_Volume
Hypothesis: Breakouts from 20-period Donchian channels on 4h, filtered by 12h EMA50 trend and volume spikes, capture strong momentum moves. The 4h timeframe balances trade frequency and signal quality, targeting 20-50 trades/year to minimize fee drag while capturing trends in both bull and bear markets.
"""

name = "4h_Donchian20_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 4h data for Donchian channels and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 20-period Donchian channels on 4h
    # Upper band: highest high of last 20 periods
    # Lower band: lowest low of last 20 periods
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian (20) and EMA50 (50)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(upper[i]) or
            np.isnan(lower[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs 12h EMA50
        uptrend_12h = close[i] > ema50_12h_aligned[i]
        downtrend_12h = close[i] < ema50_12h_aligned[i]
        
        # Volume filter
        volume_filter = volume[i] > vol_ema20[i] * 1.5
        
        if position == 0:
            # Long breakout: price breaks above upper Donchian band with volume and uptrend
            if high[i] > upper[i] and uptrend_12h and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below lower Donchian band with volume and downtrend
            elif low[i] < lower[i] and downtrend_12h and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below lower Donchian band or trend fails
            if low[i] < lower[i] or not uptrend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above upper Donchian band or trend fails
            if high[i] > upper[i] or not downtrend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals