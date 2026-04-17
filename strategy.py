#!/usr/bin/env python3
"""
4h Donchian Breakout + 1d Volume Spike + 1w Trend Filter
Long: Price breaks above Donchian(20) high + 1d volume > 2x 20-period average + 1w EMA50 rising
Short: Price breaks below Donchian(20) low + 1d volume > 2x 20-period average + 1w EMA50 falling
Exit: Opposite Donchian breakout or volume drops below average
Uses price channel breakouts with volume confirmation and higher timeframe trend filter.
Designed to capture strong trending moves while avoiding choppy markets.
Target: 80-160 total trades over 4 years (20-40/year)
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
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1d volume SMA(20)
    vol_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    # Calculate 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 20  # need Donchian channels
    
    for i in range(start_idx, n):
        if (np.isnan(vol_sma_20_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1d = volume_1d[i] if i < len(volume_1d) else volume_1d[-1]  # fallback for alignment
        vol_sma_val = vol_sma_20_1d_aligned[i]
        ema_50_val = ema_50_1w_aligned[i]
        ema_50_prev = ema_50_1w_aligned[i-1] if i > 0 else ema_50_val
        highest = highest_high[i]
        lowest = lowest_low[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high + volume spike + 1w EMA rising
            if price > highest and vol_1d > 2.0 * vol_sma_val and ema_50_val > ema_50_prev:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + volume spike + 1w EMA falling
            elif price < lowest and vol_1d > 2.0 * vol_sma_val and ema_50_val < ema_50_prev:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below Donchian low or volume drops below average
            if price < lowest or vol_1d < vol_sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price breaks above Donchian high or volume drops below average
            if price > highest or vol_1d < vol_sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_1dVolumeSpike_1wTrend"
timeframe = "4h"
leverage = 1.0