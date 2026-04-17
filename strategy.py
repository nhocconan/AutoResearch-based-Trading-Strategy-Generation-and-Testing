#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with daily Camarilla pivot breakout + volume confirmation + 1d ADX trend filter.
Long when price breaks above daily R1 with volume > 1.5x 20-period average and ADX > 25.
Short when price breaks below daily S1 with volume > 1.5x 20-period average and ADX > 25.
Daily Camarilla pivots capture key intraday institutional levels; breakouts with volume and trend filter reduce false signals.
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag. Uses discrete sizing 0.25.
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
    
    # Get daily data for Camarilla pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels (R1, S1)
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    # R1 = Pivot + Range * 1.1 / 2
    # S1 = Pivot - Range * 1.1 / 2
    r1_1d = pivot_1d + range_1d * 0.55
    s1_1d = pivot_1d - range_1d * 0.55
    
    # Calculate daily ADX (14-period)
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
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
    
    # Get 1d volume 20-period average
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 12h
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for ADX and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above daily R1 with volume and trend
            if (close[i] > r1_1d_aligned[i] and 
                volume_confirmed and 
                trend_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily S1 with volume and trend
            elif (close[i] < s1_1d_aligned[i] and 
                  volume_confirmed and 
                  trend_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below daily pivot or trend weakens
            pivot_1d = (high_1d + low_1d + close_1d) / 3
            pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
            if (close[i] < pivot_1d_aligned[i] or 
                adx_aligned[i] < 20):  # exit when trend weakens
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above daily pivot or trend weakens
            pivot_1d = (high_1d + low_1d + close_1d) / 3
            pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
            if (close[i] > pivot_1d_aligned[i] or 
                adx_aligned[i] < 20):  # exit when trend weakens
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dCamarillaR1S1_Volume_ADX"
timeframe = "12h"
leverage = 1.0