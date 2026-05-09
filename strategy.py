#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_TRIX_VolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for TRIX calculation and chop filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # TRIX(12) on weekly close
    close_1w = df_1w['close'].values
    ema1 = pd.Series(close_1w).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix_raw = np.where(ema3[:-2] != 0, (ema3[3:] - ema3[:-3]) / ema3[:-2] * 100, 0)
    trix_full = np.concatenate([np.full(3, np.nan), trix_raw])
    trix_1w = trix_full[:len(df_1w)]
    
    # Align TRIX to 12h
    trix_1w_aligned = align_htf_to_ltf(prices, df_1w, trix_1w)
    
    # Weekly Choppiness Index (CHOP) for regime filter
    atr_1w = []
    for i in range(1, len(df_1w)):
        tr = max(
            df_1w['high'].iloc[i] - df_1w['low'].iloc[i],
            abs(df_1w['high'].iloc[i] - df_1w['close'].iloc[i-1]),
            abs(df_1w['low'].iloc[i] - df_1w['close'].iloc[i-1])
        )
        atr_1w.append(tr)
    atr_1w = np.concatenate([[np.nan], atr_1w])
    
    sum_tr = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(df_1w['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1w['low']).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(sum_tr / (highest_high - lowest_low)) / np.log10(14)
    chop_1w = chop_raw
    
    # Align CHOP to 12h
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Volume spike detection (12h timeframe)
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_1w_aligned[i]) or np.isnan(chop_1w_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma20[i]  # Volume spike
        chop_filter = chop_1w_aligned[i] > 50.0  # Only trade in choppy/neutral regimes
        
        if position == 0:
            # Long: TRIX crosses above zero with volume spike in choppy market
            if i > 0 and trix_1w_aligned[i] > 0 and trix_1w_aligned[i-1] <= 0 and vol_ok and chop_filter:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume spike in choppy market
            elif i > 0 and trix_1w_aligned[i] < 0 and trix_1w_aligned[i-1] >= 0 and vol_ok and chop_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below zero or volatility breaks down
            if trix_1w_aligned[i] < 0 or volume[i] < 0.5 * vol_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero or volatility breaks down
            if trix_1w_aligned[i] > 0 or volume[i] < 0.5 * vol_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals