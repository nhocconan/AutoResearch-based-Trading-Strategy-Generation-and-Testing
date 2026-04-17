#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1w Donchian(20) breakout + volume confirmation + ADX trend filter.
Long when price breaks above weekly Donchian high with volume > 1.5x 20-period average and ADX > 25.
Short when price breaks below weekly Donchian low with volume > 1.5x 20-period average and ADX > 25.
Weekly Donchian channels provide robust structure for 12h timeframe; breakouts with volume and trend filter reduce false signals.
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag. Uses discrete sizing 0.25.
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
    
    # Get weekly data for Donchian channels, volume, and ADX
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly Donchian channels (20-period)
    high_series = pd.Series(high_1w)
    low_series = pd.Series(low_1w)
    upper_20 = high_series.rolling(window=20, min_periods=20).max().values
    lower_20 = low_series.rolling(window=20, min_periods=20).min().values
    
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
    
    # Align all to 12h
    upper_20_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    volume_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for ADX and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_1w_aligned[i]) or 
            np.isnan(volume_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current weekly volume > 1.5x 20-period average
        volume_confirmed = volume_1w_aligned[i] > 1.5 * vol_ma_20_1w_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with volume and trend
            if (close[i] > upper_20_aligned[i] and 
                volume_confirmed and 
                trend_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low with volume and trend
            elif (close[i] < lower_20_aligned[i] and 
                  volume_confirmed and 
                  trend_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below weekly Donchian midpoint or trend weakens
            midpoint_20 = (upper_20 + lower_20) / 2
            midpoint_20_aligned = align_htf_to_ltf(prices, df_1w, midpoint_20)
            if (close[i] < midpoint_20_aligned[i] or 
                adx_aligned[i] < 20):  # exit when trend weakens
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above weekly Donchian midpoint or trend weakens
            midpoint_20 = (upper_20 + lower_20) / 2
            midpoint_20_aligned = align_htf_to_ltf(prices, df_1w, midpoint_20)
            if (close[i] > midpoint_20_aligned[i] or 
                adx_aligned[i] < 20):  # exit when trend weakens
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1wDonchian20_Volume_ADX"
timeframe = "12h"
leverage = 1.0