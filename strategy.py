#!/usr/bin/env python3
"""
4h_1d_Donchian20_Breakout_Volume_Confirmation_v2
Hypothesis: 4-hour Donchian(20) breakouts with 1-day volume confirmation and ATR volatility filter.
In bull markets: breakouts above upper band signal continuation. 
In bear markets: breakdowns below lower band signal continuation.
Volume filter ensures breakouts have institutional participation.
ATR filter avoids low-volatility false breakouts.
Designed for low trade frequency (target: 20-50/year) to minimize fee drag.
Works in both bull and bear markets by capturing directional momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for volume average
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate daily average volume (20-day)
    volume_daily = df_daily['volume'].values
    vol_series = pd.Series(volume_daily)
    vol_avg_daily = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_avg_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_daily)
    
    # Main timeframe data (4h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # ATR (14-period) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_avg_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ok = volume[i] > vol_avg_daily_aligned[i]
        vol_filter = vol_ok and (atr[i] > 0)  # Ensure volatility present
        
        if position == 0:
            # Long breakout: price breaks above upper Donchian band with volume
            if price > upper[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below lower Donchian band with volume
            elif price < lower[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to middle of channel or breaks below lower band
            mid = (upper[i] + lower[i]) / 2.0
            if price < mid or price < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle of channel or breaks above upper band
            mid = (upper[i] + lower[i]) / 2.0
            if price > mid or price > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Donchian20_Breakout_Volume_Confirmation_v2"
timeframe = "4h"
leverage = 1.0