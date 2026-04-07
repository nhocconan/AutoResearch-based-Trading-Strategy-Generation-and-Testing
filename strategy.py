#!/usr/bin/env python3
"""
4h_donchian_breakout_1d_trend_volume_v1
Hypothesis: 4-hour Donchian channel breakouts with daily trend filter and volume confirmation capture momentum in both bull and bear markets. The daily trend filter ensures trades align with higher timeframe direction, reducing false breakouts. Volume confirmation adds conviction. Targets 20-50 trades per year by requiring confluence of three factors: price breakout, volume surge, and daily trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for Donchian channel
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:  # Need at least 20 periods for Donchian
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels on 4h
    # Upper band: highest high over last 20 periods
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over last 20 periods
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate EMA50 on daily for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    
    # Align 4h indicators to 4h timeframe (already aligned, but using helper for consistency)
    donchian_upper = align_htf_to_ltf(prices, df_4h, high_20)
    donchian_lower = align_htf_to_ltf(prices, df_4h, low_20)
    ema50_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 20-period volume average on 4h
    vol_sma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(ema50_4h[i]) or 
            np.isnan(vol_sma_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x average volume
        vol_confirm = volume[i] > 2.0 * vol_sma_4h[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian lower OR trend turns down
            if close[i] < donchian_lower[i] or close[i] < ema50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper OR trend turns up
            if close[i] > donchian_upper[i] or close[i] > ema50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above Donchian upper + volume + uptrend
            if (close[i] > donchian_upper[i] and 
                vol_confirm and 
                close[i] > ema50_4h[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian lower + volume + downtrend
            elif (close[i] < donchian_lower[i] and 
                  vol_confirm and 
                  close[i] < ema50_4h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals