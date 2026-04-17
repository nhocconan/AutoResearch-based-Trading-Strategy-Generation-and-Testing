#!/usr/bin/env python3
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
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    
    # Use previous day's pivots (avoid look-ahead)
    r1_1d_prev = np.roll(r1_1d, 1)
    s1_1d_prev = np.roll(s1_1d, 1)
    r2_1d_prev = np.roll(r2_1d, 1)
    s2_1d_prev = np.roll(s2_1d, 1)
    r1_1d_prev[0] = np.nan
    s1_1d_prev[0] = np.nan
    r2_1d_prev[0] = np.nan
    s2_1d_prev[0] = np.nan
    
    # Align daily pivot levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_1d_prev)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_1d_prev)
    r2_4h = align_htf_to_ltf(prices, df_1d, r2_1d_prev)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2_1d_prev)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR filter to avoid low volatility environments
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma10 = pd.Series(atr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # Need pivots, volume MA20, ATR MA10
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_ma10[i]) or 
            np.isnan(r1_4h[i]) or 
            np.isnan(s1_4h[i]) or
            np.isnan(r2_4h[i]) or
            np.isnan(s2_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        # Volatility filter: ATR > ATR MA10 (avoid low volatility)
        volatility_filter = atr[i] > atr_ma10[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and volatility
            if (close[i] > r1_4h[i] and volume_filter and volatility_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and volatility
            elif (close[i] < s1_4h[i] and volume_filter and volatility_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below pivot point
            if close[i] < pivot_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above pivot point
            if close[i] > pivot_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DailyPivot_R1S1_Breakout_Volume_Filter"
timeframe = "4h"
leverage = 1.0