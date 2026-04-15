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
    
    # Get 12h HTF data once before loop (primary HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period) - primary trend structure
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    upper_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian to 12h timeframe (no alignment needed for same TF)
    upper_20_12h_aligned = upper_20_12h
    lower_20_12h_aligned = lower_20_12h
    
    # Get 1d HTF data for weekly pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week (using 1d data)
    # Weekly high/low/close from 5 trading days ago (prior week)
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(5).values
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(5).values
    weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().shift(5).values
    
    # Weekly pivot: (H+L+C)/3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly R1: 2*P - L
    weekly_r1 = 2 * weekly_pivot - weekly_low
    # Weekly S1: 2*P - H
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Get 1w HTF data for regime filter (choppiness)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w ATR for choppiness indicator
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr3 = np.abs(low_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # True range for choppiness calculation
    true_range_1w = tr_1w
    
    # Sum of true ranges over 14 periods
    sum_tr_14 = pd.Series(true_range_1w).rolling(window=14, min_periods=14).sum().values
    
    # Max high - min low over 14 periods
    max_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Choppiness Index: 100 * log10(sum(tr14)/range14) / log10(14)
    chop = 100 * np.log10(sum_tr_14 / (range_14 + 1e-10)) / np.log10(14)
    
    # Align all HTF indicators to 12h timeframe
    upper_20_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_20_12h)
    lower_20_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_20_12h)
    weekly_pivot_12h = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_12h = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_12h = align_htf_to_ltf(prices, df_1d, weekly_s1)
    chop_12h = align_htf_to_ltf(prices, df_1w, chop, additional_delay_bars=0)
    
    # Calculate 12h ATR(14) for volatility filter
    tr1_12h = high - low
    tr2_12h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_12h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr_14_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Precompute session filter (00-24 UTC for 12h - less restrictive)
    hours = prices.index.hour
    in_session = (hours >= 0) & (hours <= 23)  # Always true for 12h, kept for structure
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_12h_aligned[i]) or np.isnan(lower_20_12h_aligned[i]) or 
            np.isnan(weekly_pivot_12h[i]) or np.isnan(weekly_r1_12h[i]) or 
            np.isnan(weekly_s1_12h[i]) or np.isnan(atr_14_12h[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(chop_12h[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 12h price breaks above 12h Donchian upper (20) - bullish breakout
        # 2. Price above weekly pivot (bullish bias from prior week)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Choppiness regime: CHOP > 61.8 (ranging market) for mean reversion edge
        if (close[i] > upper_20_12h_aligned[i] and
            close[i] > weekly_pivot_12h[i] and
            volume_ratio[i] > 1.5 and
            chop_12h[i] > 61.8):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 12h price breaks below 12h Donchian lower (20) - bearish breakdown
        # 2. Price below weekly pivot (bearish bias from prior week)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Choppiness regime: CHOP > 61.8 (ranging market) for mean reversion edge
        elif (close[i] < lower_20_12h_aligned[i] and
              close[i] < weekly_pivot_12h[i] and
              volume_ratio[i] > 1.5 and
              chop_12h[i] > 61.8):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_12h_Donchian20_1d_WeeklyPivot_Volume_Chop_Filter_v1"
timeframe = "12h"
leverage = 1.0