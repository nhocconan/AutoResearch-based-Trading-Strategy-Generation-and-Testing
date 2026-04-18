# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_Adaptive_Pivot_Regime_Breakout
6h strategy using weekly pivot regime and daily volume confirmation with adaptive position sizing.
Long: Price breaks above weekly R2 + volume > 1.5x daily avg + weekly trend up
Short: Price breaks below weekly S2 + volume > 1.5x daily avg + weekly trend down
Exit: Opposite breakout or trend reversal
Designed for 15-30 trades/year per symbol (60-120 total over 4 years)
Works in bull markets (breakout continuation) and bear markets (breakdown continuation)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot regime and trend
    df_1w = get_htf_data(prices, '1w')
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (using prior week's data)
    # Pivot = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # R2 = Pivot + (H - L)
    r2_1w = pivot_1w + (high_1w - low_1w)
    # S2 = Pivot - (H - L)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Weekly EMA50 and EMA200 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all weekly data to 6h timeframe
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Align daily volume to 6h timeframe
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # need enough for weekly EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r2_1w_aligned[i]) or np.isnan(s2_1w_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend conditions
        uptrend = ema_50_1w_aligned[i] > ema_200_1w_aligned[i]
        downtrend = ema_50_1w_aligned[i] < ema_200_1w_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > r2_1w_aligned[i]
        breakdown_down = close[i] < s2_1w_aligned[i]
        
        if position == 0:
            # Long: uptrend + volume + breakout above weekly R2
            if uptrend and vol_confirm and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + breakdown below weekly S2
            elif downtrend and vol_confirm and breakdown_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change, volume confirmation, or breakdown below weekly S2
            if not uptrend or (vol_confirm and breakdown_down):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change, volume confirmation, or breakout above weekly R2
            if not downtrend or (vol_confirm and breakout_up):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Adaptive_Pivot_Regime_Breakout"
timeframe = "6h"
leverage = 1.0