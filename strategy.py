#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R mean reversion with 1d ADX regime filter and volume confirmation.
Williams %R identifies overbought/oversold conditions, ADX filter ensures trades align with trending markets (ADX>25),
volume spike confirms momentum. Designed for 6h timeframe to reduce trade frequency while capturing reversals in trends.
Works in both bull and bear markets by using ADX to filter trend strength and Williams %R for mean reversion entries.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
Uses discrete position sizing (0.25) to balance return and fee drag.
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
    
    # Calculate 1d ADX for regime filter (trending market: ADX > 25)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/period)
    def WilderSmoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # first value is simple average
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    tr_smooth = WilderSmoothing(tr, period)
    dm_plus_smooth = WilderSmoothing(dm_plus, period)
    dm_minus_smooth = WilderSmoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = WilderSmoothing(dx, period)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h Williams %R for mean reversion
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_6h) / (highest_high - lowest_low) * -100
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Calculate volume spike: current volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need ADX and volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: trending market (ADX > 25)
        is_trending = adx_aligned[i] > 25
        
        # Williams %R conditions
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        if position == 0:
            # Long: Williams %R oversold AND trending market AND volume spike
            if oversold and is_trending and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought AND trending market AND volume spike
            elif overbought and is_trending and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral territory (-50) or opposite extreme
            exit_signal = False
            if position == 1:
                # Exit long when Williams %R rises above -50 (mean reversion complete)
                if williams_r_aligned[i] > -50:
                    exit_signal = True
            elif position == -1:
                # Exit short when Williams %R falls below -50 (mean reversion complete)
                if williams_r_aligned[i] < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_MeanReversion_1dADX_Regime_VolumeSpike"
timeframe = "6h"
leverage = 1.0