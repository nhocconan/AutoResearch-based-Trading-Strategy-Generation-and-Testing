#!/usr/bin/env python3
"""
Hypothesis: 4h price breaking above/below 1-week Donchian Channel (15) with volume above 1.4x 50-period average and 1-week ADX > 25.
Trades in direction of weekly trend to avoid counter-trend whipsaws. Uses weekly timeframe for trend filtering.
Target: 15-25 trades/year per symbol (60-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-week Donchian Channel (15-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 15:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Upper and lower bands
    upper_dc = pd.Series(high_1w).rolling(window=15, min_periods=15).max().values
    lower_dc = pd.Series(low_1w).rolling(window=15, min_periods=15).min().values
    
    # Middle line for trend
    mid_dc = (upper_dc + lower_dc) / 2
    
    # Calculate 1-week ADX (14-period)
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / (tr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_1w = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 50-period average volume
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Get aligned indicators
        upper_dc_aligned = align_htf_to_ltf(prices, df_1w, upper_dc)[i]
        lower_dc_aligned = align_htf_to_ltf(prices, df_1w, lower_dc)[i]
        mid_dc_aligned = align_htf_to_ltf(prices, df_1w, mid_dc)[i]
        adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)[i]
        vol_ma_50_aligned = vol_ma_50[i]  # already LTF
        
        # Check for NaN values
        if (np.isnan(upper_dc_aligned) or np.isnan(lower_dc_aligned) or 
            np.isnan(mid_dc_aligned) or np.isnan(adx_1w_aligned) or 
            np.isnan(vol_ma_50_aligned)):
            continue
        
        # Volume confirmation (> 1.4x average)
        volume_confirm = volume[i] > 1.4 * vol_ma_50_aligned
        
        # ADX trend filter (> 25)
        trend_filter = adx_1w_aligned > 25
        
        if position == 0:  # No position - look for entries
            if volume_confirm and trend_filter:
                # Long: price breaks above upper DC and trend is up (above mid)
                if close[i] > upper_dc_aligned and close[i-1] <= upper_dc_aligned and close[i] > mid_dc_aligned:
                    position = 1
                    signals[i] = position_size
                # Short: price breaks below lower DC and trend is down (below mid)
                elif close[i] < lower_dc_aligned and close[i-1] >= lower_dc_aligned and close[i] < mid_dc_aligned:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:  # Long position - exit when price breaks below lower DC
            if close[i] < lower_dc_aligned and close[i-1] >= lower_dc_aligned:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit when price breaks above upper DC
            if close[i] > upper_dc_aligned and close[i-1] <= upper_dc_aligned:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_1wDC15_1wADX25_Volume"
timeframe = "4h"
leverage = 1.0