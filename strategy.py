#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Camarilla_Pivot_Volume_Regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Get 1w data for Chop filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Camarilla pivot levels from 1d OHLC
    # Pivot = (H + L + C) / 3
    pivot = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    r1 = df_1d['close'] + (df_1d['high'] - df_1d['low']) * 1.1 / 12
    s1 = df_1d['close'] - (df_1d['high'] - df_1d['low']) * 1.1 / 12
    
    # Calculate Chop(14) on 1w high/low/close
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_1w.sum() / (highest_high - lowest_low)) / np.log10(14)
    
    # Align indicators to 1d timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop, additional_delay_bars=0)
    
    # Volume average for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
            
        # Chop filter: avoid trending markets (Chop < 38.2) and extreme chop (Chop > 61.8)
        # Trade only in moderate chop: 38.2 <= Chop <= 61.8
        chop_filter = (chop_aligned[i] >= 38.2) and (chop_aligned[i] <= 61.8)
        
        # Volume confirmation: current volume > 1.5x average
        volume_filter = volume[i] > 1.5 * vol_avg[i]
        
        if position == 0:
            # Long when price crosses above S1 with volume and chop filter
            if (close[i] > s1_aligned[i] and 
                chop_filter and 
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short when price crosses below R1 with volume and chop filter
            elif (close[i] < r1_aligned[i] and 
                  chop_filter and 
                  volume_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price crosses below pivot
            if close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price crosses above pivot
            if close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals