#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d ADX regime filter + volume confirmation.
Long when price breaks above Donchian(20) high, 1d ADX > 25 (trending), and volume > 1.5x 20-period average.
Short when price breaks below Donchian(20) low, 1d ADX > 25, and volume > 1.5x 20-period average.
Exit when price crosses Donchian(20) midpoint or ADX < 20 (range regime).
Uses 1d for ADX regime filter, 4h for price action and volume.
Target: 20-50 total trades over 4 years (5-12.5/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing for ATR
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Wilder's smoothing for DI
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        return adx
    
    # Calculate 1d ADX
    adx_14 = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Calculate 4h volume average (20-period)
    volume_ma = np.full(n, np.nan)
    for i in range(20-1, n):
        volume_ma[i] = np.mean(volume[i-20+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_14_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (ADX > 25)
        is_trending = adx_14_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Donchian breakout levels
        donchian_high = highest_high[i]
        donchian_low = lowest_low[i]
        donchian_mid = (donchian_high + donchian_low) / 2
        
        if position == 0:
            # Long: price breaks above Donchian high + trending + volume confirmation
            if close[i] > donchian_high and is_trending and volume_confirmed:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian low + trending + volume confirmation
            elif close[i] < donchian_low and is_trending and volume_confirmed:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint OR regime becomes ranging (ADX < 20)
            if close[i] < donchian_mid or adx_14_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint OR regime becomes ranging (ADX < 20)
            if close[i] > donchian_mid or adx_14_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_1dADX_Volume_Regime"
timeframe = "4h"
leverage = 1.0