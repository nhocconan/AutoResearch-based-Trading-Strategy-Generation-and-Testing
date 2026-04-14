#!/usr/bin/env python3
"""
Hypothesis: 12-hour price breaking above/below 1-day Donchian Channel (20) with volume above 1.3x 20-period average and 1-day ADX > 25.
Trades in direction of 1-day trend to avoid counter-trend whipsaws. Uses Donchian period (20) and ADX threshold (25) for balanced signal frequency.
Target: 15-25 trades/year per symbol (60-100 total over 4 years).
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
    
    # Calculate 1-day Donchian Channel (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Upper and lower bands
    upper_dc = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_dc = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Middle line for trend
    mid_dc = (upper_dc + lower_dc) / 2
    
    # Calculate 1-day ADX (14-period)
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
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
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-period average volume (20 periods of 12h = 10 days)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Get aligned indicators
        upper_dc_aligned = align_htf_to_ltf(prices, df_1d, upper_dc)[i]
        lower_dc_aligned = align_htf_to_ltf(prices, df_1d, lower_dc)[i]
        mid_dc_aligned = align_htf_to_ltf(prices, df_1d, mid_dc)[i]
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)[i]
        vol_ma_20_aligned = vol_ma_20[i]  # already LTF
        
        # Check for NaN values
        if (np.isnan(upper_dc_aligned) or np.isnan(lower_dc_aligned) or 
            np.isnan(mid_dc_aligned) or np.isnan(adx_1d_aligned) or 
            np.isnan(vol_ma_20_aligned)):
            continue
        
        # Volume confirmation (> 1.3x average)
        volume_confirm = volume[i] > 1.3 * vol_ma_20_aligned
        
        # ADX trend filter (> 25)
        trend_filter = adx_1d_aligned > 25
        
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

name = "12h_1dDC20_1dADX25_Volume"
timeframe = "12h"
leverage = 1.0