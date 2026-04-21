#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_VolumeFilter_V1
Hypothesis: 6h Donchian(20) breakouts in direction of weekly pivot bias with volume confirmation work for BTC/ETH in both bull and bear markets. Uses 1d for Donchian calculation and 1w for weekly pivot (bullish if close > weekly pivot, bearish if <). Target: 12-37 trades/year per symbol (50-150 over 4 years). Discrete sizing 0.25 minimizes fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data once for Donchian channels (6h entry timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Load weekly data once for pivot bias (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Daily Donchian(20) for breakout signals
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Weekly pivot: (weekly high + weekly low + weekly close) / 3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Volume filter: 24-period average (approx 6 days on 6h)
    vol_ma = prices['volume'].rolling(window=24, min_periods=24).mean().values
    
    # ATR for stoploss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation
        volume_ok = volume > 1.5 * vol_ma[i]
        
        # Weekly pivot bias
        bullish_bias = price > weekly_pivot_aligned[i]
        bearish_bias = price < weekly_pivot_aligned[i]
        
        if position == 0:
            # Long: price breaks above daily Donchian high in bullish bias with volume
            if bullish_bias and volume_ok:
                if price > donchian_high_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Short: price breaks below daily Donchian low in bearish bias with volume
            elif bearish_bias and volume_ok:
                if price < donchian_low_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: price reaches daily Donchian low or stoploss
            if price <= donchian_low_aligned[i] or price < prices['close'].iloc[i-1] - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price reaches daily Donchian high or stoploss
            if price >= donchian_high_aligned[i] or price > prices['close'].iloc[i-1] + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_VolumeFilter_V1"
timeframe = "6h"
leverage = 1.0