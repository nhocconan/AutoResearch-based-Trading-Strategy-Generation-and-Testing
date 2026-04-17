#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1-week and 1-day structure.
Trade weekly Donchian breakouts aligned with daily EMA50 trend filter and volume confirmation.
Use 12h for precise entry timing to maintain low trade frequency (15-30/year).
In bull markets: follow weekly trend via breakouts above daily EMA50.
In bear markets: avoid counter-trend trades by requiring alignment with weekly trend.
Weekly structure provides robustness; daily filter prevents whipsaws.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for structure (Donchian channels)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels (20-period)
    high_max_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly and daily data to 12h
    high_max_20_aligned = align_htf_to_ltf(prices, df_1w, high_max_20)
    low_min_20_aligned = align_htf_to_ltf(prices, df_1w, low_min_20)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.5x 24-period average (to avoid noise)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max_20_aligned[i]) or np.isnan(low_min_20_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with volume and above daily EMA50
            if close[i] > high_max_20_aligned[i] and volume_filter[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low with volume and below daily EMA50
            elif close[i] < low_min_20_aligned[i] and volume_filter[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below weekly Donchian low (mean reversion)
            if close[i] < low_min_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above weekly Donchian high (mean reversion)
            if close[i] > high_max_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1wDonchian20_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0