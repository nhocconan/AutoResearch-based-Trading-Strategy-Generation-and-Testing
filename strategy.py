#!/usr/bin/env python3
name = "1d_Trix_15_ZeroCross_Volume_ChopFilter"
timeframe = "1d"
leverage = 1.0

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
    
    # TRIX: 15-period EMA of EMA of EMA of price
    ema1 = pd.Series(close).ewm(span=15, adjust=False).mean()
    ema2 = ema1.ewm(span=15, adjust=False).mean()
    ema3 = ema2.ewm(span=15, adjust=False).mean()
    trix = 100 * (ema3.diff() / ema3.shift(1))
    trix = trix.fillna(0).values
    
    # 1-week EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: volume > 1.5x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20 * 1.5
    
    # Choppiness Index (14-period): range-bound detection
    atr14 = []
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.insert(tr, 0, high[0] - low[0])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr.sum(axis=0) / (hh - ll)) / np.log10(14) if False else \
        100 * np.log10(pd.Series(atr).rolling(14, min_periods=14).sum() / (hh - ll)) / np.log10(14)
    chop = chop.fillna(50).values
    chop_range = chop > 61.8  # chop > 61.8 = ranging market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(trix[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(volume_ok[i]) or np.isnan(chop_range[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        trix_cross_up = trix[i] > 0 and trix[i-1] <= 0
        trix_cross_down = trix[i] < 0 and trix[i-1] >= 0
        
        if position == 0:
            # Long: TRIX crosses above zero + above weekly EMA34 + volume + ranging market
            if trix_cross_up and close[i] > ema34_1w_aligned[i] and volume_ok[i] and chop_range[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero + below weekly EMA34 + volume + ranging market
            elif trix_cross_down and close[i] < ema34_1w_aligned[i] and volume_ok[i] and chop_range[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit on opposite TRIX cross
            if position == 1 and trix_cross_down:
                signals[i] = 0.0
                position = 0
            elif position == -1 and trix_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals