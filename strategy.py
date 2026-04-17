#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with volume confirmation and 1d ADX trend filter.
Long when price breaks above 20-period Donchian high with volume > 2.0x 12h avg volume AND 1d ADX > 25.
Short when price breaks below 20-period Donchian low with volume > 2.0x 12h avg volume AND 1d ADX > 25.
Exit when price touches the opposite Donchian level (long exits at lower band, short exits at upper band).
Uses 12h for execution and volume, 1d for ADX trend filter.
Donchian channels provide clear structure, volume confirms institutional interest, ADX ensures trending markets.
Designed to capture sustained moves in both bull and bear markets while avoiding choppy regimes.
Target: 15-25 trades/year per symbol.
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
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(x, period):
        x = x.copy()
        alpha = 1.0 / period
        for i in range(1, len(x)):
            if np.isnan(x[i-1]):
                x[i] = x[i]
            else:
                x[i] = alpha * x[i] + (1 - alpha) * x[i-1]
        return x
    
    period = 14
    tr_period = 100  # enough for warmup
    if len(tr) >= tr_period:
        atr = wilders_smoothing(tr, period)
        plus_di = 100 * wilders_smoothing(plus_dm, period) / atr
        minus_di = 100 * wilders_smoothing(minus_dm, period) / atr
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = wilders_smoothing(dx, period)
    else:
        adx = np.full_like(tr, np.nan)
    
    # Align 1d ADX to primary timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    adx_trending = adx_aligned > 25
    
    # Get 12h data for execution and volume
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    donch_high = rolling_max(high_12h, 20)
    donch_low = rolling_min(low_12h, 20)
    
    # Calculate 12h volume MA (20-period)
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h indicators to primary timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_trending[i]) or 
            np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 2.0x 20-bar average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20_aligned[i]
        
        # Breakout conditions
        breakout_upper = close[i] > donch_high_aligned[i]
        breakout_lower = close[i] < donch_low_aligned[i]
        
        if position == 0:
            # Long: break above upper Donchian with volume confirmation and ADX trending
            if breakout_upper and volume_confirmed and adx_trending[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume confirmation and ADX trending
            elif breakout_lower and volume_confirmed and adx_trending[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch lower Donchian band
            if close[i] <= donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch upper Donchian band
            if close[i] >= donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Volume_1dADX_Trend"
timeframe = "12h"
leverage = 1.0