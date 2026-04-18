#!/usr/bin/env python3
"""
12h Donchian Breakout with Volume Confirmation and 1w Trend Filter
Hypothesis: Donchian(20) breakouts on 12h chart capture significant moves. Volume confirmation ensures institutional participation.
1-week EMA50 filter ensures we only trade in the direction of the higher timeframe trend, working in both bull and bear markets.
Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1h data for Donchian channels (more responsive than 12h for calculation)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels on 1h data (20-period high/low)
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    
    donchian_high = np.full_like(high_1h, np.nan)
    donchian_low = np.full_like(low_1h, np.nan)
    
    for i in range(len(high_1h)):
        if i < 19:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high_1h[i-19:i+1])
            donchian_low[i] = np.min(low_1h[i-19:i+1])
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1h, donchian_low)
    
    # Volume confirmation: current volume > 2.0x 30-period average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i < 29:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-29:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    # Get 1-week data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = np.full_like(close_1w, np.nan)
    
    # Calculate EMA50 on weekly close
    alpha = 2.0 / (50 + 1)
    for i in range(len(close_1w)):
        if i == 0:
            ema_50_1w[i] = close_1w[i]
        elif np.isnan(ema_50_1w[i-1]):
            ema_50_1w[i] = close_1w[i]
        else:
            ema_50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_50_1w[i-1]
    
    # Align EMA50 to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume spike and above weekly EMA50
            if close[i] > donchian_high_aligned[i] and vol_spike[i] and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume spike and below weekly EMA50
            elif close[i] < donchian_low_aligned[i] and vol_spike[i] and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below Donchian low or volume dies
            if close[i] < donchian_low_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above Donchian high or volume dies
            if close[i] > donchian_high_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_Volume_1wTrend"
timeframe = "12h"
leverage = 1.0