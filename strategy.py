#!/usr/bin/env python3
"""
1d_SR_Trend_Breakout_V1
Strategy: Daily support/resistance breakout with 1-week trend filter and volume confirmation.
Long: Price breaks above 20-day high in uptrend with volume spike.
Short: Price breaks below 20-day low in downtrend with volume spike.
Designed for 1d timeframe: ~10-20 trades/year per symbol (40-80 total over 4 years).
Works in bull/bear via trend filter and breakout logic with volume confirmation.
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
    
    # Get weekly data for trend filter (1w)
    df_1w = get_htf_data(prices, '1w')
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA to daily timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get daily data for support/resistance levels
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 20-day high and low (Donchian channels)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all daily data to daily timeframe (no alignment needed for same timeframe)
    # But we still use the function for consistency and proper handling
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # need enough for weekly EMA34 and daily lookbacks
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_aligned[i]) or np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend condition: price relative to weekly EMA34
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 2.0 * vol_ma_aligned[i]
        
        # Breakout conditions
        breakout_up = high[i] > high_20_aligned[i]
        breakout_down = low[i] < low_20_aligned[i]
        
        if position == 0:
            # Long: uptrend + volume + breakout up
            if uptrend and vol_confirm and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + breakout down
            elif downtrend and vol_confirm and breakout_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change, or breakdown below 20-day low
            if not uptrend or low[i] < low_20_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change, or breakout above 20-day high
            if not downtrend or high[i] > high_20_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_SR_Trend_Breakout_V1"
timeframe = "1d"
leverage = 1.0