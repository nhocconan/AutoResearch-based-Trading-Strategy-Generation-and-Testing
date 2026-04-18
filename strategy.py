#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout with Volume Confirmation and Trend Filter
Hypothesis: Weekly Donchian channels (20-period) on 1d timeframe capture major trend breaks.
Price breaking above weekly high or below weekly low with volume confirmation indicates
strong momentum. Trend filter using 1w EMA ensures we only trade in direction of higher timeframe trend.
Works in bull markets (buy breakouts) and bear markets (sell breakdowns) by following price action.
Designed for low frequency to minimize fee drag: target 10-30 trades/year.
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
    
    # Get weekly data for trend filter and Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = np.zeros_like(close_1w)
    for i in range(len(close_1w)):
        if i < 20:
            ema_1w[i] = np.mean(close_1w[0:i+1])
        else:
            ema_1w[i] = ema_1w[i-1] * 0.9 + close_1w[i] * 0.1  # EMA approximation
    
    # Align weekly EMA to daily timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    donchian_high = np.zeros_like(high_1w)
    donchian_low = np.zeros_like(low_1w)
    
    for i in range(len(high_1w)):
        if i < 19:
            donchian_high[i] = np.max(high_1w[0:i+1]) if i >= 0 else high_1w[i]
            donchian_low[i] = np.min(low_1w[0:i+1]) if i >= 0 else low_1w[i]
        else:
            donchian_high[i] = np.max(high_1w[i-19:i+1])
            donchian_low[i] = np.min(low_1w[i-19:i+1])
    
    # Align Donchian channels to daily timeframe (use previous week's levels)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup for weekly calculations
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with volume spike and above weekly EMA
            if close[i] > donchian_high_aligned[i] and vol_spike[i] and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low with volume spike and below weekly EMA
            elif close[i] < donchian_low_aligned[i] and vol_spike[i] and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below weekly Donchian low or trend changes
            if close[i] < donchian_low_aligned[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above weekly Donchian high or trend changes
            if close[i] > donchian_high_aligned[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Donchian_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0