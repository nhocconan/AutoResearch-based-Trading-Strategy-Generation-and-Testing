#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 1d volume confirmation and 1d ADX trend filter, plus exit on opposite breakout.
Uses 4h primary timeframe for balanced trade frequency, 1d for trend/volume filters to reduce noise.
Aims for 20-50 trades/year (80-200 total over 4 years) with discrete sizing to minimize fee drag.
Works in bull via breakouts, in bear via short breakdowns, avoids whipsaws with trend filter.
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
    
    # 4h Donchian channels (20-period)
    def donchian_channels(high, low, window):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        for i in range(window-1, len(high)):
            upper[i] = np.max(high[i-window+1:i+1])
            lower[i] = np.min(low[i-window+1:i+1])
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channels(high, low, 20)
    
    # 1d data for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # 1d ADX (14-period)
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
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(30, n):
        # Get aligned 1d indicators
        vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)[i]
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx)[i]
        
        # Check for NaN values
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma_1d_aligned) or np.isnan(adx_aligned)):
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma_1d_aligned
        
        # ADX trend filter (> 20)
        trend_filter = adx_aligned > 20
        
        if position == 0:  # No position - look for entries
            if volume_confirm and trend_filter:
                # Long: break above Donchian upper
                if close[i] > donchian_upper[i]:
                    position = 1
                    signals[i] = position_size
                # Short: break below Donchian lower
                elif close[i] < donchian_lower[i]:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:  # Long position - exit when price breaks below Donchian lower
            if close[i] < donchian_lower[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit when price breaks above Donchian upper
            if close[i] > donchian_upper[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian_1dVol_ADX"
timeframe = "4h"
leverage = 1.0