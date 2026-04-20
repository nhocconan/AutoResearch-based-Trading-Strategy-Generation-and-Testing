#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_WeeklyTrend_6h_Pullback"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # === Weekly Trend Filter (200-day EMA on daily) ===
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === 6h Pullback Logic ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h EMA21 for pullback entry
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation (20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema200_aligned[i]) or np.isnan(ema21[i]) or 
            np.isnan(vol_ma20[i]) or np.isnan(close[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        ema200_val = ema200_aligned[i]
        ema21_val = ema21[i]
        vol_ratio = volume[i] / vol_ma20[i] if vol_ma20[i] > 0 else 0
        
        if position == 0:
            # Long: above weekly EMA200, pullback to EMA21 with volume
            if close_val > ema200_val and close_val <= ema21_val * 1.005 and vol_ratio > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: below weekly EMA200, bounce to EMA21 with volume
            elif close_val < ema200_val and close_val >= ema21_val * 0.995 and vol_ratio > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: break below EMA21 or reverse below weekly EMA200
            if close_val < ema21_val or close_val < ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: break above EMA21 or reverse above weekly EMA200
            if close_val > ema21_val or close_val > ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals