#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Choppiness_Donchian_Breakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    # Get 1d data (already loaded, but we need it for calculations)
    # For 1w data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w: Choppiness Index (regime filter) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # ATR(14)
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = 100 * np.log10(sum_atr_14 / (hh_1w - ll_1w)) / np.log10(14)
    chop = np.where((hh_1w - ll_1w) > 0, chop, 50)  # Avoid division by zero
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # === 1d: Donchian Channel (breakout) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20)
    dc_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dc_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Get values
        chop_val = chop_1w_aligned[i]
        dc_high_val = dc_high[i]
        dc_low_val = dc_low[i]
        close_val = close[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(chop_val) or np.isnan(dc_high_val) or np.isnan(dc_low_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout + low chop (trending) + volume
            if (close_val > dc_high_val and  # Breakout above Donchian high
                chop_val < 38.2 and          # Trending regime (chop < 38.2)
                vol_ratio_val > 1.5):        # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown + low chop (trending) + volume
            elif (close_val < dc_low_val and  # Breakdown below Donchian low
                  chop_val < 38.2 and         # Trending regime
                  vol_ratio_val > 1.5):       # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Donchian breakdown OR high chop (ranging)
            if (close_val < dc_low_val or  # Breakdown below Donchian low
                chop_val > 61.8):          # Ranging regime (chop > 61.8)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Donchian breakout OR high chop (ranging)
            if (close_val > dc_high_val or  # Breakout above Donchian high
                chop_val > 61.8):           # Ranging regime
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals