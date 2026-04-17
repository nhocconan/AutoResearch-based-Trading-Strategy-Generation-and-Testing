#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot (using last completed week)
    # Week high/low from 7 trading days (approx)
    high_7d = pd.Series(high_1w).rolling(window=7, min_periods=7).max().values
    low_7d = pd.Series(low_1w).rolling(window=7, min_periods=7).min().values
    close_prev = pd.Series(close_1w).shift(1).values  # Previous week close
    
    # Standard pivot point formula
    pivot = (high_7d + low_7d + close_prev) / 3.0
    r1 = 2 * pivot - low_7d
    s1 = 2 * pivot - high_7d
    r2 = pivot + (high_7d - low_7d)
    s2 = pivot - (high_7d - low_7d)
    
    # Align weekly pivots to 12h timeframe
    pivot_12h = align_htf_to_ltf(prices, df_1w, pivot)
    r1_12h = align_htf_to_ltf(prices, df_1w, r1)
    s1_12h = align_htf_to_ltf(prices, df_1w, s1)
    r2_12h = align_htf_to_ltf(prices, df_1w, r2)
    s2_12h = align_htf_to_ltf(prices, df_1w, s2)
    
    # Volume confirmation (20-period MA on 12h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness regime filter (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) for chop
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d)
    tr2 = pd.Series(high_1d) - pd.Series(low_1d).shift(1)
    tr3 = pd.Series(close_1d).shift(1) - pd.Series(close_1d).shift(2)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14)/(HH14-LL14)) / log10(14)
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr14 / (hh_14 - ll_14 + 1e-10)) / np.log10(14)
    
    # Align chop to 12h timeframe
    chop_12h = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 14)  # volume MA20 and chop period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(pivot_12h[i]) or 
            np.isnan(r1_12h[i]) or 
            np.isnan(s1_12h[i]) or 
            np.isnan(r2_12h[i]) or 
            np.isnan(s2_12h[i]) or 
            np.isnan(chop_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Chop filter: chop > 50 indicates ranging market (good for mean reversion at pivot levels)
        chop_filter = chop_12h[i] > 50
        
        if position == 0:
            # Long: break above R2 with volume in choppy market
            if close[i] > r2_12h[i] and volume_filter and chop_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below S2 with volume in choppy market
            elif close[i] < s2_12h[i] and volume_filter and chop_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below R1 or chop drops (trending) or volume dries up
            if close[i] < r1_12h[i] or chop_12h[i] < 40 or volume[i] < volume_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above S1 or chop drops (trending) or volume dries up
            if close[i] > s1_12h[i] or chop_12h[i] < 40 or volume[i] < volume_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyPivot_R2_S2_Breakout_Volume_Chop"
timeframe = "12h"
leverage = 1.0