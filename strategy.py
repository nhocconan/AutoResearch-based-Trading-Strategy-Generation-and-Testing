#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 34:
        return np.zeros(n)
    
    # Load 1d data once for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Choppiness Index (14-period)
    atr_1d = np.zeros(len(df_1d))
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.abs(high_1d[0] - low_1d[0])], tr])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr_14 / (max_hh - min_ll)) / np.log10(14)
    chop = np.where((max_hh - min_ll) == 0, 50, chop)  # avoid division by zero
    
    chop_align = align_htf_to_ltf(prices, df_1d, chop)
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Price channels: Donchian(20) on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    upper_donch = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_donch = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    upper_donch_aligned = align_htf_to_ltf(prices, df_12h, upper_donch)
    lower_donch_aligned = align_htf_to_ltf(prices, df_12h, lower_donch)
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any data is not ready
        if (np.isnan(chop_align[i]) or 
            np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(upper_donch_aligned[i]) or 
            np.isnan(lower_donch_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        chop_val = chop_align[i]
        ema50 = ema50_12h_aligned[i]
        upper = upper_donch_aligned[i]
        lower = lower_donch_aligned[i]
        
        # Regime filter: Choppiness > 61.8 = ranging market (mean revert)
        ranging = chop_val > 61.8
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks below lower Donchian + ranging + volume spike
            if price < lower and ranging and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks above upper Donchian + ranging + volume spike
            elif price > upper and ranging and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to middle of channel or volatility breaks down
            mid = (upper + lower) / 2
            exit_signal = False
            
            if position == 1:  # long position
                if price > mid:
                    exit_signal = True
            
            elif position == -1:  # short position
                if price < mid:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_ChopRange_Volume"
timeframe = "12h"
leverage = 1.0