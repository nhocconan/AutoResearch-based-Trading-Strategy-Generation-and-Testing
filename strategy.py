#!/usr/bin/env python3
"""
Hypothesis: 4h price breaking above/below 12-hour Donchian channel (15-period) with volume above 1.4x 20-period average and 12-hour ADX > 20.
Trades in direction of 12-hour trend to avoid counter-trend whipsaws. Uses Donchian for clear breakout signals.
Targets 25-35 trades/year per symbol (100-140 total over 4 years).
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
    
    # Calculate 12-hour Donchian channel (15-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 15:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper band: highest high over 15 periods
    upper_band = pd.Series(high_12h).rolling(window=15, min_periods=15).max().values
    # Lower band: lowest low over 15 periods
    lower_band = pd.Series(low_12h).rolling(window=15, min_periods=15).min().values
    
    # Calculate 12-hour ADX (10-period)
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_10 = pd.Series(tr).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    dm_plus_10 = pd.Series(dm_plus).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    dm_minus_10 = pd.Series(dm_minus).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_10 / (tr_10 + 1e-10)
    di_minus = 100 * dm_minus_10 / (tr_10 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_12h = pd.Series(dx).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    
    # Calculate 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Get aligned indicators
        upper_band_aligned = align_htf_to_ltf(prices, df_12h, upper_band)[i]
        lower_band_aligned = align_htf_to_ltf(prices, df_12h, lower_band)[i]
        adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)[i]
        vol_ma_20_aligned = vol_ma_20[i]  # already LTF
        
        # Check for NaN values
        if (np.isnan(upper_band_aligned) or np.isnan(lower_band_aligned) or 
            np.isnan(adx_12h_aligned) or np.isnan(vol_ma_20_aligned)):
            continue
        
        # Volume confirmation (> 1.4x average)
        volume_confirm = volume[i] > 1.4 * vol_ma_20_aligned
        
        # ADX trend filter (> 20)
        trend_filter = adx_12h_aligned > 20
        
        if position == 0:  # No position - look for entries
            if volume_confirm and trend_filter:
                # Long: price breaks above upper Donchian band
                if close[i] > upper_band_aligned and close[i-1] <= upper_band_aligned:
                    position = 1
                    signals[i] = position_size
                # Short: price breaks below lower Donchian band
                elif close[i] < lower_band_aligned and close[i-1] >= lower_band_aligned:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:  # Long position - exit when price breaks below lower band
            if close[i] < lower_band_aligned and close[i-1] >= lower_band_aligned:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit when price breaks above upper band
            if close[i] > upper_band_aligned and close[i-1] <= upper_band_aligned:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_12hDonchian15_12hADX20_Volume_v1"
timeframe = "4h"
leverage = 1.0