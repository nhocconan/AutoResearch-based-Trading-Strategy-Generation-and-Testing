#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and ADX regime filter.
Long when price breaks above 20-period Donchian high with above-average volume and ADX > 25.
Short when price breaks below 20-period Donchian low with above-average volume and ADX > 25.
Exit when price returns to the midpoint of the Donchian channel.
Uses 1d for volume and ADX regime, 12h for price action.
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
    
    # Get 1d data for volume and ADX regime
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
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
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        return adx
    
    # Calculate 1d volume SMA (20-period)
    vol_sma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators
    adx_14 = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    vol_sma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20)
    
    # Calculate 12h Donchian channel (20-period)
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        middle = (upper + lower) / 2
        return upper, lower, middle
    
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_14_aligned[i]) or 
            np.isnan(vol_sma_20_aligned[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(donchian_middle[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 20-period SMA
        vol_ratio = vol_1d[i // 12] / vol_sma_20_aligned[i] if vol_sma_20_aligned[i] > 0 else 0
        # Note: 12 12h bars = 1d bar, so we use integer division to get 1d index
        
        # ADX regime filter: only trade when trending (ADX > 25)
        is_trending = adx_14_aligned[i] > 25
        
        # Donchian breakout conditions
        price = close[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper with volume confirmation and trending regime
            if price > donchian_upper[i] and vol_ratio > 1.0 and is_trending:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower with volume confirmation and trending regime
            elif price < donchian_lower[i] and vol_ratio > 1.0 and is_trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to Donchian middle
            if price < donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to Donchian middle
            if price > donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dVolume_ADX_Trend"
timeframe = "12h"
leverage = 1.0