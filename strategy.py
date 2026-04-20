#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for weekly context
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ATR for volatility normalization
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[tr1[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Weekly trend via EMA(8) on weekly closes
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_8 = pd.Series(close_1w).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_8_aligned = align_htf_to_ltf(prices, df_1w, ema_8)
    
    # Daily Donchian channel (20-period)
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_8_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND price above weekly EMA
            if price > donch_high_aligned[i] and price > ema_8_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND price below weekly EMA
            elif price < donch_low_aligned[i] and price < ema_8_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low OR price below weekly EMA
            if price < donch_low_aligned[i] or price < ema_8_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high OR price above weekly EMA
            if price > donch_high_aligned[i] or price > ema_8_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian_WeeklyEMA_Filter_v1"
timeframe = "1d"
leverage = 1.0