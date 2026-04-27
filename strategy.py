#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_TripleFilter
Hypothesis: Donchian(20) breakouts with volume confirmation (2x 20-period average), 
trend filter (12h EMA50), and ADX(14) > 25 for trend strength capture sustained moves.
Works in bull (breakouts continue) and bear (breakdowns continue) by using 
breakout direction with trend alignment. Target: 15-30 trades/year per symbol.
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Donchian(20) channels
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ADX(14) for trend strength filter
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], closed[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.concatenate([[0], high[1:] - high[:-1]])
    down_move = np.concatenate([[0], low[:-1] - low[1:]])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and DX
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 20  # need 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume, trend, and ADX > 25
            if (close[i] > high_max[i] and volume_spike[i] and 
                close[i] > ema50_12h_aligned[i] and adx[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume, trend, and ADX > 25
            elif (close[i] < low_min[i] and volume_spike[i] and 
                  close[i] < ema50_12h_aligned[i] and adx[i] > 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns below Donchian low or trend fails
            if (close[i] < low_min[i] or 
                close[i] < ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns above Donchian high or trend fails
            if (close[i] > high_max[i] or 
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_TripleFilter"
timeframe = "4h"
leverage = 1.0