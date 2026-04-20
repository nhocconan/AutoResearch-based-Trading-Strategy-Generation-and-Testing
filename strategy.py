#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_KAMA_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # === 1d: KAMA trend direction ===
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will fix below
    # Recompute volatility properly
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i == 0:
            er[i] = 0
        else:
            change_sum = np.sum(change[max(0, i-9):i+1]) if i >= 1 else change[i]
            volatility_sum = np.sum(volatility[max(0, i-9):i+1]) if i >= 1 else volatility[i]
            er[i] = change_sum / volatility_sum if volatility_sum > 0 else 0
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # === 1w: Higher timeframe trend filter ===
    close_1w = df_1w['close'].values
    # Simple trend: price above/below 20-period EMA
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === 4h: Price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume spike detector (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align HTF indicators to 4h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if np.isnan(kama_aligned[i]) or np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_val = kama_aligned[i]
        ema20_1w_val = ema20_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        close_val = close[i]
        
        if position == 0:
            # Long: Price above KAMA (1d trend up) AND above weekly EMA20 (long-term up) AND volume spike
            if (close_val > kama_val and 
                close_val > ema20_1w_val and 
                vol_ratio_val > 2.0):  # Strong volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA (1d trend down) AND below weekly EMA20 (long-term down) AND volume spike
            elif (close_val < kama_val and 
                  close_val < ema20_1w_val and 
                  vol_ratio_val > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend reversal or loss of momentum
            if (close_val < kama_val or      # Price below KAMA (trend change)
                vol_ratio_val < 1.0):        # Volume dried up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or loss of momentum
            if (close_val > kama_val or      # Price above KAMA (trend change)
                vol_ratio_val < 1.0):        # Volume dried up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals