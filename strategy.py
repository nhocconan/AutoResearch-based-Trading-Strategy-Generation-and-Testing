#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian channel breakout with volume confirmation and 1w ADX trend filter.
Long when price breaks above 20-bar Donchian high with volume > 2.0x 12h avg volume AND 1w ADX > 25.
Short when price breaks below 20-bar Donchian low with volume > 2.0x 12h avg volume AND 1w ADX > 25.
Exit when price touches the opposite Donchian band or ADX < 20 (trend weakening).
Uses 12h for execution and volume, 1w for ADX trend filter.
Designed to capture strong trends with volume confirmation, avoiding choppy markets.
Target: 12-30 trades/year per symbol.
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
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ADX(14)
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align indices
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+,
    period = 14
    tr_period = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_plus_period = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_minus_period = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_period / tr_period
    di_minus = 100 * dm_minus_period / tr_period
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    adx_trending = adx > 25
    adx_weakening = adx < 20
    adx_trending[0] = False
    adx_weakening[0] = False
    
    # Get 12h data for execution and volume
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume MA (20-period)
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align 1w ADX and 12h indicators to primary timeframe
    adx_trending_aligned = align_htf_to_ltf(prices, df_1w, adx_trending)
    adx_weakening_aligned = align_htf_to_ltf(prices, df_1w, adx_weakening)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_trending_aligned[i]) or 
            np.isnan(adx_weakening_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 2.0x 20-bar average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20_aligned[i]
        
        # Breakout conditions
        breakout_high = close[i] > donchian_high_aligned[i]
        breakout_low = close[i] < donchian_low_aligned[i]
        
        # Exit conditions
        exit_long = (close[i] < donchian_low_aligned[i]) or adx_weakening_aligned[i]
        exit_short = (close[i] > donchian_high_aligned[i]) or adx_weakening_aligned[i]
        
        if position == 0:
            # Long: break above Donchian high with volume confirmation and ADX trending
            if (breakout_high and volume_confirmed and adx_trending_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume confirmation and ADX trending
            elif (breakout_low and volume_confirmed and adx_trending_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch Donchian low or ADX weakening
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch Donchian high or ADX weakening
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Volume_1wADX_Trend"
timeframe = "12h"
leverage = 1.0