#!/usr/bin/env python3
"""
4h_TRIX_Volume_Spike_Chop
Hypothesis: TRIX (triple smoothed EMA) crosses zero with volume spike and choppiness regime filter captures momentum in trending markets while avoiding whipsaws in ranging conditions. TRIX filters noise better than MACD, volume confirms conviction, and choppiness regime ensures we only trade in trending markets (CHOP < 38.2). Works in both bull and bear markets by following the trend direction as defined by TRIX zero-cross and 1w EMA filter.
Target: 50-150 total trades over 4 years (12-38/year).
"""
name = "4h_TRIX_Volume_Spike_Chop"
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
    
    # TRIX (15-period triple EMA)
    ema1 = pd.Series(close).ewm(span=15, adjust=False).mean()
    ema2 = ema1.ewm(span=15, adjust=False).mean()
    ema3 = ema2.ewm(span=15, adjust=False).mean()
    trix = 100 * (ema3 / ema3.shift(1) - 1)
    trix = trix.values
    
    # Choppiness Index (14-period) - regime filter
    def true_range(high, low, prev_close):
        return np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10(sum_tr14 / (max_hh - min_ll)) / np.log10(14)
    chop = np.where((max_hh - min_ll) == 0, 50, chop)  # avoid division by zero
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_avg * 2.0)
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 15)  # Need TRIX and CHOP warmed up
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(trix[i]) or np.isnan(chop[i]) or np.isnan(vol_avg[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero + volume spike + trending regime (CHOP < 38.2) + price above 1w EMA50
            if trix[i] > 0 and trix[i-1] <= 0 and volume_spike[i] and chop[i] < 38.2 and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero + volume spike + trending regime (CHOP < 38.2) + price below 1w EMA50
            elif trix[i] < 0 and trix[i-1] >= 0 and volume_spike[i] and chop[i] < 38.2 and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: TRIX crosses zero in opposite direction
            if position == 1:
                if trix[i] < 0 and trix[i-1] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if trix[i] > 0 and trix[i-1] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals