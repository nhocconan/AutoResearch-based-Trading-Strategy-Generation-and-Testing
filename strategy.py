#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 12h Camarilla R1/S1 breakout + volume confirmation + ADX trend filter.
Long when price breaks above 12h Camarilla R1 with volume > 1.5x 20-period average and ADX > 25.
Short when price breaks below 12h Camarilla S1 with volume > 1.5x 20-period average and ADX > 25.
Camarilla pivot levels from 12h provide intraday structure; breakouts with volume and trend filter reduce false signals.
Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag. Uses discrete sizing 0.25.
Works in both bull (trend continuation) and bear (mean reversion after volatility spikes).
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
    
    # Get 12h data for Camarilla levels, volume, and ADX
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h Camarilla levels (R1, S1)
    # Pivot = (high + low + close) / 3
    # Range = high - low
    # R1 = close + Range * 1.1 / 12
    # S1 = close - Range * 1.1 / 12
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    r1_12h = close_12h + range_12h * 1.1 / 12.0
    s1_12h = close_12h - range_12h * 1.1 / 12.0
    
    # Calculate 12h ADX (14-period)
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Get 12h volume 20-period average
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 4h
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for ADX and Camarilla
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_12h_aligned[i]) or 
            np.isnan(volume_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_confirmed = volume_12h_aligned[i] > 1.5 * vol_ma_20_12h_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above 12h Camarilla R1 with volume and trend
            if (close[i] > r1_12h_aligned[i] and 
                volume_confirmed and 
                trend_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Camarilla S1 with volume and trend
            elif (close[i] < s1_12h_aligned[i] and 
                  volume_confirmed and 
                  trend_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 12h Camarilla pivot or trend weakens
            if (close[i] < pivot_12h_aligned[i] or 
                adx_aligned[i] < 20):  # exit when trend weakens
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 12h Camarilla pivot or trend weakens
            if (close[i] > pivot_12h_aligned[i] or 
                adx_aligned[i] < 20):  # exit when trend weakens
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12hCamarillaR1S1_Volume_ADX"
timeframe = "4h"
leverage = 1.0