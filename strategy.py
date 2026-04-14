#!/usr/bin/env python3
"""
Hypothesis: 1h price breaking above/below 4-hour Donchian Channel (20) with volume above 1.3x 20-period average and 4-hour ADX > 20.
Trades in direction of 4-hour trend to avoid counter-trend whipsaws. Uses Donchian period (20) and ADX threshold (20) for balanced signal frequency.
Target: 25-35 trades/year per symbol (100-140 total over 4 years).
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
    
    # Calculate 4-hour Donchian Channel (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Upper and lower bands
    upper_dc = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_dc = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Middle line for trend
    mid_dc = (upper_dc + lower_dc) / 2
    
    # Calculate 4-hour ADX (14-period)
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
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
    adx_4h = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20
    
    for i in range(50, n):
        # Get aligned indicators
        upper_dc_aligned = align_htf_to_ltf(prices, df_4h, upper_dc)[i]
        lower_dc_aligned = align_htf_to_ltf(prices, df_4h, lower_dc)[i]
        mid_dc_aligned = align_htf_to_ltf(prices, df_4h, mid_dc)[i]
        adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)[i]
        vol_ma_20_aligned = vol_ma_20[i]  # already LTF
        
        # Check for NaN values
        if (np.isnan(upper_dc_aligned) or np.isnan(lower_dc_aligned) or 
            np.isnan(mid_dc_aligned) or np.isnan(adx_4h_aligned) or 
            np.isnan(vol_ma_20_aligned)):
            continue
        
        # Volume confirmation (> 1.3x average)
        volume_confirm = volume[i] > 1.3 * vol_ma_20_aligned
        
        # ADX trend filter (> 20)
        trend_filter = adx_4h_aligned > 20
        
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

name = "1h_4hDC20_4hADX20_Volume"
timeframe = "1h"
leverage = 1.0