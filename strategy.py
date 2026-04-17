#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d Donchian channel breakout + 12h volume confirmation + ADX trend filter.
Long when price breaks above 20-period 1d Donchian high with above-average 12h volume and ADX > 25.
Short when price breaks below 20-period 1d Donchian low with above-average 12h volume and ADX > 25.
Exit when price returns to the midpoint of the 1d Donchian channel or ADX falls below 20.
Donchian channels provide clear structural breakouts that work in both bull and bear markets,
while volume confirmation reduces false breakouts and ADX filters ranging markets.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 1d data for Donchian and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian Channel (20)
    period20_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high = period20_high
    donchian_low = period20_low
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1d ADX (14)
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    avg_volume_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 6h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    avg_volume_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_volume_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(volume_12h_aligned[i]) or np.isnan(avg_volume_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 20-period average
        volume_confirmed = volume_12h_aligned[i] > avg_volume_12h_aligned[i]
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and trend
            if (close[i] > donchian_high_aligned[i] and 
                volume_confirmed and 
                strong_trend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and trend
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_confirmed and 
                  strong_trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to midpoint or trend weakens
            if (close[i] < donchian_mid_aligned[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to midpoint or trend weakens
            if (close[i] > donchian_mid_aligned[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dDonchian20_Volume_12h_ADX"
timeframe = "6h"
leverage = 1.0