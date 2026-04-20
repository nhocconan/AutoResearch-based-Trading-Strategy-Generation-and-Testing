#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Donchian channels (20 periods)
    highest_high_20w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_low_20w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    width_20w = highest_high_20w - lowest_low_20w
    
    # Align weekly Donchian to 12h timeframe
    highest_high_20w_aligned = align_htf_to_ltf(prices, df_1w, highest_high_20w)
    lowest_low_20w_aligned = align_htf_to_ltf(prices, df_1w, lowest_low_20w)
    width_20w_aligned = align_htf_to_ltf(prices, df_1w, width_20w)
    
    # Daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    volume_ma_10_1d = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    volume_ma_10_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_10_1d)
    
    # 12h price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN
        if np.isnan(highest_high_20w_aligned[i]) or np.isnan(lowest_low_20w_aligned[i]) or np.isnan(width_20w_aligned[i]) or np.isnan(volume_ma_10_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hh_20w = highest_high_20w_aligned[i]
        ll_20w = lowest_low_20w_aligned[i]
        width_20w_val = width_20w_aligned[i]
        vol_ma_val = volume_ma_10_1d_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: current volume above 10-day average
        vol_filter = vol > vol_ma_val
        
        # Volatility filter: Donchian width > 20-period median width (avoid low volatility chop)
        if i >= 30:
            width_lookback = width_20w_aligned[max(0, i-29):i+1]
            width_median = np.nanmedian(width_lookback)
            vol_filter = vol_filter and (width_20w_val > width_median)
        
        if position == 0:
            # Long: break above weekly Donchian high with volume
            if price > hh_20w and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly Donchian low with volume
            elif price < ll_20w and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to weekly Donchian midpoint or volatility collapse
            midpoint = (hh_20w + ll_20w) / 2
            if price < midpoint or (i >= 30 and width_20w_val < 0.5 * width_median):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to weekly Donchian midpoint or volatility collapse
            midpoint = (hh_20w + ll_20w) / 2
            if price > midpoint or (i >= 30 and width_20w_val < 0.5 * width_median):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyDonchianBreakout_VolumeFilter"
timeframe = "12h"
leverage = 1.0