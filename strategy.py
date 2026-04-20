#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h chart with 1d ATR-based Donchian breakout + volume confirmation + chop filter.
# Long when price breaks above 1d Donchian(20) high with volume > 1.5x avg and chop > 61.8 (ranging).
# Short when price breaks below 1d Donchian(20) low with volume > 1.5x avg and chop > 61.8.
# Uses 1d chop filter to avoid trending markets where breakouts fail.
# Target: 15-30 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Donchian channels and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Donchian channels (20-period)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # 1d Chop index (14-period)
    atr_14 = pd.Series(high_1d - low_1d).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(high_1d - low_1d).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr_14 / (atr_14 * 14)) / np.log10(14)
    
    # Align to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 12h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        high_20_val = high_20_aligned[i]
        low_20_val = low_20_aligned[i]
        chop_val = chop_aligned[i]
        vol_ok = vol_filter[i]
        
        # Chop filter: > 61.8 indicates ranging market (good for breakout fade? but we use for breakout continuation in ranging?)
        # Actually, chop > 61.8 = ranging, chop < 38.2 = trending
        # We want breakouts in ranging markets? No, breakouts work better in trending.
        # Let's reverse: chop < 38.2 = trending (good for breakout)
        trending = chop_val < 38.2
        
        if position == 0:
            # Long: price breaks above Donchian high, trending market, volume
            if price > high_20_val and trending and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, trending market, volume
            elif price < low_20_val and trending and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or chop increases (rangy)
            if price < low_20_val or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or chop increases (rangy)
            if price > high_20_val or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Donchian20_ChopFilter_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0