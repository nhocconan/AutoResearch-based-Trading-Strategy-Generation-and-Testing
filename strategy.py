#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with weekly Camarilla pivot breakout + volume confirmation + 1d EMA trend filter.
Long when price breaks above weekly R4 with volume > 1.5x 20-period average and 1d EMA50 > EMA200.
Short when price breaks below weekly S4 with volume > 1.5x 20-period average and 1d EMA50 < EMA200.
Weekly Camarilla pivots capture major institutional levels; breakouts with volume and trend filter reduce false signals.
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
    
    # Get weekly data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels
    # Pivot = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    # R4 = Pivot + (Range * 1.1/2) * 2 = Pivot + Range * 1.1
    # S4 = Pivot - (Range * 1.1/2) * 2 = Pivot - Range * 1.1
    r4_1w = pivot_1w + range_1w * 1.1
    s4_1w = pivot_1w - range_1w * 1.1
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 and EMA200
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 1d volume 20-period average
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 6h
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # need enough for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly R4 with volume and bullish trend (EMA50 > EMA200)
            if (close[i] > r4_1w_aligned[i] and 
                volume_confirmed and 
                ema50_1d_aligned[i] > ema200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S4 with volume and bearish trend (EMA50 < EMA200)
            elif (close[i] < s4_1w_aligned[i] and 
                  volume_confirmed and 
                  ema50_1d_aligned[i] < ema200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below weekly pivot or trend turns bearish
            # Use weekly pivot as dynamic exit
            # Recalculate weekly pivot for exit condition
            pivot_1w_exit = (high_1w + low_1w + close_1w) / 3
            pivot_1w_aligned_exit = align_htf_to_ltf(prices, df_1w, pivot_1w_exit)
            if (close[i] < pivot_1w_aligned_exit[i] or 
                ema50_1d_aligned[i] < ema200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above weekly pivot or trend turns bullish
            pivot_1w_exit = (high_1w + low_1w + close_1w) / 3
            pivot_1w_aligned_exit = align_htf_to_ltf(prices, df_1w, pivot_1w_exit)
            if (close[i] > pivot_1w_aligned_exit[i] or 
                ema50_1d_aligned[i] > ema200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1wCamarillaR4S4_Volume_1dEMA"
timeframe = "6h"
leverage = 1.0