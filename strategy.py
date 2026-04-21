#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly and daily data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 20 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === Weekly Donchian Channel (20-period) for trend direction ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    upper_20w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_20w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian to 6h timeframe
    upper_20w_aligned = align_htf_to_ltf(prices, df_1w, upper_20w)
    lower_20w_aligned = align_htf_to_ltf(prices, df_1w, lower_20w)
    
    # === Daily ATR for volatility filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # ATR ratio: current ATR / 20-period average ATR
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_14 / atr_ma_20
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(upper_20w_aligned[i]) or np.isnan(lower_20w_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        upper_w = upper_20w_aligned[i]
        lower_w = lower_20w_aligned[i]
        atr_ratio_val = atr_ratio_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long: price above weekly upper Donchian with volume and moderate volatility
            if (price_close > upper_w and 
                vol_ratio_val > 1.3 and 
                atr_ratio_val > 0.7 and atr_ratio_val < 2.2):
                signals[i] = 0.25
                position = 1
            # Enter short: price below weekly lower Donchian with volume and moderate volatility
            elif (price_close < lower_w and 
                  vol_ratio_val > 1.3 and 
                  atr_ratio_val > 0.7 and atr_ratio_val < 2.2):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: reverse breakout or volatility extreme
            if position == 1 and (price_close < lower_w or atr_ratio_val > 2.8 or atr_ratio_val < 0.5):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > upper_w or atr_ratio_val > 2.8 or atr_ratio_val < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WeeklyDonchian_Breakout_Volume_ATR_Filter"
timeframe = "6h"
leverage = 1.0