#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d ADX regime filter + volume confirmation.
Long when price breaks above Donchian upper band in trending regime (ADX > 25) with above-average volume.
Short when price breaks below Donchian lower band in trending regime with above-average volume.
Exit when price reverts to Donchian middle band or regime shifts to range (ADX < 20).
Uses 1d for ADX regime, 12h for price action and volume.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 1d data for regime filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period) with proper Wilder's smoothing
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
        
        # Wilder's smoothing (equivalent to RMA)
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        return adx
    
    # Calculate 1d ADX
    adx_14 = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate 12h Donchian channels (20-period)
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        middle = (upper + lower) / 2
        return upper, lower, middle
    
    upper_20, lower_20, middle_20 = calculate_donchian(high, low, 20)
    
    # Calculate 12h volume average (20-period) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_14_aligned[i]) or 
            np.isnan(upper_20[i]) or 
            np.isnan(lower_20[i]) or 
            np.isnan(middle_20[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime determination
        adx_val = adx_14_aligned[i]
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        # Volume confirmation
        vol_confirm = volume[i] > volume_ma[i]
        
        if position == 0:
            # Long: price breaks above upper band in trending regime with volume confirmation
            if close[i] > upper_20[i] and is_trending and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band in trending regime with volume confirmation
            elif close[i] < lower_20[i] and is_trending and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reverts to middle band OR regime shifts to ranging
            if close[i] < middle_20[i] or is_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reverts to middle band OR regime shifts to ranging
            if close[i] > middle_20[i] or is_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dADX_Volume_Regime"
timeframe = "12h"
leverage = 1.0