#!/usr/bin/env python3
"""
12h_Donchian_Breakout_Volume_Regime_v2
Donchian(20) breakout + volume confirmation + Chop regime filter on 12h.
Uses 1d trend filter (EMA50) for bias, 12h Donchian breakouts for entries,
volume spike for confirmation, and Choppiness index to avoid whipsaw.
Designed for low trade frequency (~20-50/year) with strong edge in both bull and bear.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d EMA50 for trend bias ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 12h Donchian channels (20-period) ===
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    for i in range(n):
        if i >= 20:
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    
    # === 12h Volume confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(n):
        if i >= 20:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
    
    vol_confirm = volume > vol_ma_20 * 1.5
    
    # === 12h Choppiness Index (14-period) ===
    atr_14 = np.full_like(close, np.nan)
    for i in range(n):
        if i >= 1:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            if i == 1:
                atr_14[i] = tr
            else:
                atr_14[i] = (atr_14[i-1] * 13 + tr) / 14
    
    chop = np.full_like(close, np.nan)
    for i in range(n):
        if i >= 14:
            sum_atr = np.sum(atr_14[i-13:i+1])
            highest_high = np.max(high[i-13:i+1])
            lowest_low = np.min(low[i-13:i+1])
            if highest_high > lowest_low:
                chop[i] = 100 * np.log10(sum_atr / (highest_high - lowest_low)) / np.log10(14)
            else:
                chop[i] = 50
    
    chop_range = chop > 61.8  # ranging market
    chop_trend = chop < 38.2  # trending market
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_confirm[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above Donchian high, above 1d EMA50 (uptrend bias),
            # volume confirmation, and trending regime (CHOP < 38.2)
            if (close[i] > donchian_high[i] and 
                close[i] > ema50_1d_aligned[i] and 
                vol_confirm[i] and 
                chop_trend[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below Donchian low, below 1d EMA50 (downtrend bias),
            # volume confirmation, and trending regime (CHOP < 38.2)
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  vol_confirm[i] and 
                  chop_trend[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price breaks below Donchian low OR chop indicates ranging
            if (close[i] < donchian_low[i] or 
                chop_range[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high OR chop indicates ranging
            if (close[i] > donchian_high[i] or 
                chop_range[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_Volume_Regime_v2"
timeframe = "12h"
leverage = 1.0