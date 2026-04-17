#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with daily Camarilla pivot breakout + volume confirmation + 4h EMA trend filter.
Long when price breaks above daily R1 with volume > 1.3x 20-period average and 4h EMA34 > EMA89.
Short when price breaks below daily S1 with volume > 1.3x 20-period average and 4h EMA34 < EMA89.
Daily Camarilla pivots capture key intraday institutional levels; breakouts with volume and trend filter reduce false signals.
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag. Uses discrete sizing 0.25.
Works in both bull and bear markets by requiring volume confirmation and trend alignment.
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels (R1, S1)
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    # R1 = Pivot + (Range * 1.1/2)
    # S1 = Pivot - (Range * 1.1/2)
    r1_1d = pivot_1d + range_1d * 0.55
    s1_1d = pivot_1d - range_1d * 0.55
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA34 and EMA89
    close_4h_series = pd.Series(close_4h)
    ema34_4h = close_4h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_4h = close_4h_series.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Calculate 4h volume 20-period average
    volume_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 12h
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    ema89_4h_aligned = align_htf_to_ltf(prices, df_4h, ema89_4h)
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    volume_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 89  # need enough for EMA89
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema34_4h_aligned[i]) or np.isnan(ema89_4h_aligned[i]) or 
            np.isnan(vol_ma_20_4h_aligned[i]) or np.isnan(volume_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.3x 20-period average
        volume_confirmed = volume_4h_aligned[i] > 1.3 * vol_ma_20_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above daily R1 with volume and bullish trend (EMA34 > EMA89)
            if (close[i] > r1_1d_aligned[i] and 
                volume_confirmed and 
                ema34_4h_aligned[i] > ema89_4h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily S1 with volume and bearish trend (EMA34 < EMA89)
            elif (close[i] < s1_1d_aligned[i] and 
                  volume_confirmed and 
                  ema34_4h_aligned[i] < ema89_4h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below daily pivot or trend turns bearish
            pivot_1d_exit = (high_1d + low_1d + close_1d) / 3
            pivot_1d_aligned_exit = align_htf_to_ltf(prices, df_1d, pivot_1d_exit)
            if (close[i] < pivot_1d_aligned_exit[i] or 
                ema34_4h_aligned[i] < ema89_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above daily pivot or trend turns bullish
            pivot_1d_exit = (high_1d + low_1d + close_1d) / 3
            pivot_1d_aligned_exit = align_htf_to_ltf(prices, df_1d, pivot_1d_exit)
            if (close[i] > pivot_1d_aligned_exit[i] or 
                ema34_4h_aligned[i] > ema89_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dCamarillaR1S1_Volume_4hEMA"
timeframe = "12h"
leverage = 1.0