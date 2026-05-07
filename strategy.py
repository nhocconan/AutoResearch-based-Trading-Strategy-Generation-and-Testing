#!/usr/bin/env python3
name = "4h_TRIX_Volume_Spike_Regime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for TRIX and regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # TRIX (15-period) on 1d
    ema1 = pd.Series(df_1d['close']).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100  # percentage change
    trix = np.concatenate([[np.nan], trix])  # align with original length
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align TRIX and EMA34 to 4h
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 2.5 * 20-period average on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (vol_ma * 2.5)
    
    # Chop regime filter: avoid trending markets (use 1d chop)
    hl_range = df_1d['high'] - df_1d['low']
    atr1 = pd.Series(hl_range).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr2 = pd.Series(atr1).ewm(span=10, adjust=False, min_periods=10).mean().values
    chop = 100 * np.log10(atr2.sum() / (hl_range.sum() + 1e-10)) / np.log10(10)
    chop = pd.Series(chop).rolling(window=14, min_periods=14).mean().values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(trix_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(chop_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero + above EMA34 + volume spike + chop > 61.8 (ranging)
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and close[i] > ema34_1d_aligned[i] and volume_ok[i] and chop_aligned[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero + below EMA34 + volume spike + chop > 61.8 (ranging)
            elif trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and close[i] < ema34_1d_aligned[i] and volume_ok[i] and chop_aligned[i] > 61.8:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: TRIX crosses zero in opposite direction or closes beyond EMA34
            if position == 1:
                if trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 or close[i] < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 or close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals