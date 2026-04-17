#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d Camarilla pivot (R1/S1) breakout + volume confirmation + 1w EMA50 trend filter.
Long when price breaks above 1d Camarilla R1 with volume > 1.5x 20-period average and price > 1w EMA50.
Short when price breaks below 1d Camarilla S1 with volume > 1.5x 20-period average and price < 1w EMA50.
Uses discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
Works in bull via trend continuation and in bear via mean reversion at pivot levels.
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
    
    # Get daily data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla pivot levels (R1, S1)
    # Pivot = (high + low + close) / 3
    # Range = high - low
    # R1 = pivot + (range * 1.1 / 12)
    # S1 = pivot - (range * 1.1 / 12)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    rng = high_1d - low_1d
    r1 = pivot + (rng * 1.1 / 12.0)
    s1 = pivot - (rng * 1.1 / 12.0)
    
    # Get weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 6h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # need enough for weekly EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        # Trend filter: price relative to weekly EMA50
        above_weekly_ema = close[i] > ema_50_1w_aligned[i]
        below_weekly_ema = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R1 with volume and above weekly EMA50
            if (close[i] > r1_aligned[i] and 
                volume_confirmed and 
                above_weekly_ema):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S1 with volume and below weekly EMA50
            elif (close[i] < s1_aligned[i] and 
                  volume_confirmed and 
                  below_weekly_ema):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 1d Camarilla pivot or below weekly EMA50
            pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
            if (close[i] < pivot_aligned[i] or 
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 1d Camarilla pivot or above weekly EMA50
            pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
            if (close[i] > pivot_aligned[i] or 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dCamarilla_R1S1_Volume_1wEMA50"
timeframe = "6h"
leverage = 1.0