#!/usr/bin/env python3
"""
Hypothesis: 1d timeframe with 1w Williams %R extreme + volume confirmation + ADX trend filter.
Long when Williams %R < -80 (oversold) with volume > 1.5x 20-day average and ADX > 25 (trending).
Short when Williams %R > -20 (overbought) with volume confirmation and ADX > 25.
Exit when Williams %R returns to -50 (mean reversion to midpoint).
Williams %R identifies exhaustion points in trends; volume confirms institutional participation;
ADX ensures we trade in trending environments where mean reversion to -50 is meaningful.
Designed to work in both bull (buy oversold in uptrend) and bear (sell overbought in downtrend) markets.
Uses 1w for Williams %R to reduce noise and 1d for execution.
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
    
    # Get 1w data for Williams %R calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Williams %R(14)
    lookback = 14
    highest_high = pd.Series(high_1w).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low_1w).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close_1w) / (highest_high - lowest_low)
    
    # Calculate 1d ADX(14) for trend filter
    # ADX requires +DI, -DI, and TR
    tr1 = pd.Series(high).rolling(window=2).max() - pd.Series(low).rolling(window=2).min()
    tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    up_move = pd.Series(high).diff()
    down_move = pd.Series(low).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1w Williams %R and ADX to 1d timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for Williams %R, ADX, and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) with volume and ADX > 25 (trending)
            if (williams_r_aligned[i] < -80 and 
                volume_confirmed and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) with volume and ADX > 25
            elif (williams_r_aligned[i] > -20 and 
                  volume_confirmed and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to -50 (mean reversion)
            if williams_r_aligned[i] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to -50
            if williams_r_aligned[i] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1wWilliamsR_Extreme_Volume_ADX25_Trend"
timeframe = "1d"
leverage = 1.0