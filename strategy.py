#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1w pivot-based breakout + volume confirmation + ADX trend filter.
Long when price breaks above weekly pivot R1 with volume > 1.5x 20-period average and ADX > 25.
Short when price breaks below weekly pivot S1 with volume > 1.5x 20-period average and ADX > 25.
Weekly pivots from higher timeframe provide robust structure; breakouts with volume and trend filter reduce false signals.
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
    
    # Get weekly data for pivot points, volume, and ADX
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly pivot points (standard formula)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    
    # Calculate weekly ADX (14-period)
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
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
    
    # Get weekly volume 20-period average
    vol_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 6h
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    volume_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for ADX and pivots
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_1w_aligned[i]) or 
            np.isnan(volume_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1w volume > 1.5x 20-period average
        volume_confirmed = volume_1w_aligned[i] > 1.5 * vol_ma_20_1w_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above weekly pivot R1 with volume and trend
            if (close[i] > r1_aligned[i] and 
                volume_confirmed and 
                trend_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly pivot S1 with volume and trend
            elif (close[i] < s1_aligned[i] and 
                  volume_confirmed and 
                  trend_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below weekly pivot point or trend weakens
            if (close[i] < pivot[i] if not np.isnan(pivot[i]) else False) or  \
               (adx_aligned[i] < 20):  # exit when trend weakens
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above weekly pivot point or trend weakens
            if (close[i] > pivot[i] if not np.isnan(pivot[i]) else False) or  \
               (adx_aligned[i] < 20):  # exit when trend weakens
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1wPivot_R1S1_Volume_ADX"
timeframe = "6h"
leverage = 1.0