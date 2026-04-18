#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrendVolume_Strict_v1
Hypothesis: Trade Donchian(20) breakout on 4h with 1d trend and volume confirmation. 
Long when price breaks above 20-period high and 1d EMA50 > EMA200 and volume > 1.5x 24-bar average. 
Short when price breaks below 20-period low and 1d EMA50 < EMA200 and volume > 1.5x average.
Uses strict volume filter (1.5x) and trend filter to reduce trades to 20-40/year. 
Works in bull by catching breakouts, in bear by shorting breakdowns with trend confirmation.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 and EMA200
    ema50_1d = np.full_like(close_1d, np.nan)
    ema200_1d = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema50_1d[i] = (close_1d[i] * 2 / (50 + 1)) + (ema50_1d[i-1] * (49 / (50 + 1)))
    
    if len(close_1d) >= 200:
        ema200_1d[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema200_1d[i] = (close_1d[i] * 2 / (200 + 1)) + (ema200_1d[i-1] * (199 / (200 + 1)))
    
    # Align 1d EMAs to 4h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Donchian channels on 4h (20-period)
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    
    lookback = 20
    if len(high) >= lookback:
        for i in range(lookback - 1, len(high)):
            donchian_high[i] = np.max(high[i - lookback + 1:i + 1])
            donchian_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, vol_period, 50)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above Donchian high + 1d uptrend (EMA50 > EMA200) + volume
            if close[i] > donchian_high[i] and ema50_1d_aligned[i] > ema200_1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low + 1d downtrend (EMA50 < EMA200) + volume
            elif close[i] < donchian_low[i] and ema50_1d_aligned[i] < ema200_1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: breakdown below Donchian low or 1d trend turns down
            if close[i] < donchian_low[i] or ema50_1d_aligned[i] < ema200_1d_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: breakout above Donchian high or 1d trend turns up
            if close[i] > donchian_high[i] or ema50_1d_aligned[i] > ema200_1d_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dTrendVolume_Strict_v1"
timeframe = "4h"
leverage = 1.0