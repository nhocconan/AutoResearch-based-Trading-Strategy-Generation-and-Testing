#!/usr/bin/env python3
"""
12h_donchian20_volume_regime_v1
Hypothesis: Uses Donchian(20) breakouts on 12h timeframe with volume confirmation and 1d chop regime filter.
Long when price breaks above upper band with volume surge in trending market (CHOP < 61.8).
Short when price breaks below lower band with volume surge in trending market.
Uses 1d timeframe for chop regime filter to avoid false breakouts in sideways markets.
Targets 15-30 trades/year to minimize fee drag while capturing significant breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian20_volume_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: 1.5x 24-period average (24*12h = 12 days)
    vol_ma_period = 24
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Donchian channels (20-period) on 12h
    donchian_period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(donchian_period-1, n):
        upper[i] = np.max(high[i-donchian_period+1:i+1])
        lower[i] = np.min(low[i-donchian_period+1:i+1])
    
    # Get 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Chop Index (14-period) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                               np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]  # First value
    
    # ATR (14-period)
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        atr_1d[i] = np.mean(tr[i-13:i+1])
    
    # Chop Index = 100 * log10(sum(TR14)/(ATR14 * 14)) / log10(14)
    chop = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        if atr_1d[i] > 0:
            tr_sum = np.sum(tr[i-13:i+1])
            chop[i] = 100 * np.log10(tr_sum / (atr_1d[i] * 14)) / np.log10(14)
    
    # Trending market: CHOP < 61.8 (below this = trending)
    trending = chop < 61.8
    
    # Align chop regime to 12h timeframe
    trending_aligned = align_htf_to_ltf(prices, df_1d, trending.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(vol_ma_period, donchian_period, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_ma[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(trending_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price drops below lower band or market becomes ranging
            if close[i] < lower[i] or trending_aligned[i] < 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price rises above upper band or market becomes ranging
            if close[i] > upper[i] or trending_aligned[i] < 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above upper band, volume surge, trending market
            if (close[i] > upper[i] and 
                vol_surge[i] and 
                trending_aligned[i] > 0.5):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below lower band, volume surge, trending market
            elif (close[i] < lower[i] and 
                  vol_surge[i] and 
                  trending_aligned[i] > 0.5):
                position = -1
                signals[i] = -0.25
    
    return signals