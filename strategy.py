#!/usr/bin/env python3
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
    
    # Get daily data for Camarilla pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivot levels (H3/L3 for tighter entries)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    H3 = close_1d + (range_hl * 1.1 / 4)
    L3 = close_1d - (range_hl * 1.1 / 4)
    
    # Align pivot levels to daily timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Get weekly data for regime filter (choppiness index)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range calculation for weekly data
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w = np.concatenate([[np.nan], tr_1w])
    
    # Choppiness Index (CHOP) = 100 * log10(sum(ATR)/log(N)) / log10(N)
    atr_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    chop_1w = 100 * np.log10(sum_atr_14 / np.log10(14)) / np.log10(highest_high_14 - lowest_low_14)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34, 14)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or
            np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: current daily volume above 20-day average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume[i] > vol_ma_20[i]
        
        # Regime filter: Choppiness Index > 61.8 (ranging market) for mean reversion
        regime_filter = chop_1w_aligned[i] > 61.8
        
        # Entry conditions: Camarilla H3/L3 breakout with volume and ranging regime
        long_breakout = close[i] > H3_aligned[i]
        short_breakout = close[i] < L3_aligned[i]
        
        long_entry = uptrend and long_breakout and volume_filter and regime_filter
        short_entry = downtrend and short_breakout and volume_filter and regime_filter
        
        # Exit conditions: Close below/above pivot level (mean reversion to daily pivot)
        pivot_val = (high_1d + low_1d + close_1d) / 3
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_val)
        long_exit = close[i] < pivot_aligned[i]
        short_exit = close[i] > pivot_aligned[i]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_VolumeChop_Regime"
timeframe = "1d"
leverage = 1.0