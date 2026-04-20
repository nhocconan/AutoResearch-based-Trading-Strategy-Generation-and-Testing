#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Choppiness_Donchian_Breakout_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 300:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === Daily Choppiness Index (14) for Regime Filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14)
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of true ranges over 14 periods
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = 100 * np.log10(sum_tr14 / (atr14 * 14)) / np.log10(14)
    chop = np.where((hh14 - ll14) > 0, chop, 50)  # Avoid division by zero
    chop = np.nan_to_num(chop, nan=50.0)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === Daily Donchian Channel (20) for Breakout Signals ===
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # === 12h Price, Volume, and ATR for Position Sizing ===
    close = prices['close'].values
    volume = volumes = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # ATR(20) for 12h timeframe
    tr_12h = np.maximum(high - low, np.maximum(np.abs(high - np.concatenate([[close[0]], close[:-1]])), np.abs(low - np.concatenate([[close[0]], close[:-1]]))))
    atr_12h = pd.Series(tr_12h).rolling(window=20, min_periods=20).mean().values
    
    # Volume ratio (20-period)
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close[i]
        vol_ratio_val = vol_ratio[i]
        chop_val = chop_aligned[i]
        high_20_val = high_20_aligned[i]
        low_20_val = low_20_aligned[i]
        atr_val = atr_12h[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(chop_val) or 
            np.isnan(high_20_val) or np.isnan(low_20_val) or np.isnan(atr_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high in trending market (chop < 61.8) with volume
            if (close_val > high_20_val and 
                chop_val < 61.8 and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low in trending market (chop < 61.8) with volume
            elif (close_val < low_20_val and 
                  chop_val < 61.8 and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below Donchian low or chop increases (range) or volume dries up
            if (close_val < low_20_val or 
                chop_val > 61.8 or 
                vol_ratio_val < 0.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above Donchian high or chop increases (range) or volume dries up
            if (close_val > high_20_val or 
                chop_val > 61.8 or 
                vol_ratio_val < 0.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals