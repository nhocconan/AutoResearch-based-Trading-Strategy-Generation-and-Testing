#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d ADX regime filter + volume confirmation.
Long when price breaks above Donchian(20) high in trending regime (ADX>25) with above-average volume.
Short when price breaks below Donchian(20) low in trending regime (ADX>25) with above-average volume.
Exit on opposite Donchian(10) break or regime shift to ranging (ADX<20).
Uses 12h for price action/volume, 1d for ADX regime filter.
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
    
    # Get 12h data for Donchian channels and volume MA
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h Donchian(20) for breakout signals
    def calculate_donchian_channel(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    # Calculate 12h Donchian(10) for exit signals
    donchian_20_upper, donchian_20_lower = calculate_donchian_channel(high_12h, low_12h, 20)
    donchian_10_upper, donchian_10_lower = calculate_donchian_channel(high_12h, low_12h, 10)
    
    # Calculate 12h volume 20-period MA for confirmation
    volume_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX (14-period) for regime filter
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
    
    # Calculate 1d ADX
    adx_14 = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align all 12h and 1d indicators to 12h timeframe
    donchian_20_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_20_upper)
    donchian_20_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_20_lower)
    donchian_10_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_10_upper)
    donchian_10_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_10_lower)
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_20_upper_aligned[i]) or 
            np.isnan(donchian_20_lower_aligned[i]) or
            np.isnan(donchian_10_upper_aligned[i]) or
            np.isnan(donchian_10_lower_aligned[i]) or
            np.isnan(volume_ma_20_aligned[i]) or
            np.isnan(adx_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime determination: trending if ADX > 25, ranging if ADX < 20
        is_trending = adx_14_aligned[i] > 25
        is_ranging = adx_14_aligned[i] < 20
        
        # Volume confirmation: current volume > 20-period MA
        volume_confirm = volume[i] > volume_ma_20_aligned[i]
        
        if position == 0:
            # Long: Donchian(20) breakout up + trending regime + volume confirmation
            if close[i] > donchian_20_upper_aligned[i] and is_trending and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Donchian(20) breakout down + trending regime + volume confirmation
            elif close[i] < donchian_20_lower_aligned[i] and is_trending and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Donchian(10) breakout down OR regime shifts to ranging
            if close[i] < donchian_10_lower_aligned[i] or is_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Donchian(10) breakout up OR regime shifts to ranging
            if close[i] > donchian_10_upper_aligned[i] or is_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dADX_Volume_Regime"
timeframe = "12h"
leverage = 1.0