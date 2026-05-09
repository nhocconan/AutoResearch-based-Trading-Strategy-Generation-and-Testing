#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Choppiness_Filtered_Donchian_Breakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR and Choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate ATR(14) on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Choppiness Index on daily
    # Sum of true ranges over period
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over period
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    choppiness_raw = 100 * np.log10(sum_tr / (hh - ll)) / np.log10(14)
    # Replace inf/NaN with 50 (neutral)
    choppiness_raw = np.where((hh - ll) == 0, 50, choppiness_raw)
    choppiness_raw = np.nan_to_num(choppiness_raw, nan=50.0)
    chop = choppiness_raw
    
    # Calculate Donchian channels on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    upper_dc = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_dc = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align all indicators to 4h timeframe
    atr_1d_4h = align_htf_to_ltf(prices, df_1d, atr_1d)
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    upper_dc_4h = align_htf_to_ltf(prices, df_4h, upper_dc)
    lower_dc_4h = align_htf_to_ltf(prices, df_4h, lower_dc)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(atr_1d_4h[i]) or np.isnan(chop_4h[i]) or 
            np.isnan(upper_dc_4h[i]) or np.isnan(lower_dc_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        atr = atr_1d_4h[i]
        chop_val = chop_4h[i]
        upper_dc_val = upper_dc_4h[i]
        lower_dc_val = lower_dc_4h[i]
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr > 0.01 * close[i]  # ATR > 1% of price
        
        if position == 0:
            # Enter long when price breaks above upper Donchian in trending market (low chop)
            if close[i] > upper_dc_val and chop_val < 38.2 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short when price breaks below lower Donchian in trending market
            elif close[i] < lower_dc_val and chop_val < 38.2 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below lower Donchian or chop increases (ranging market)
            if close[i] < lower_dc_val or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above upper Donchian or chop increases
            if close[i] > upper_dc_val or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals