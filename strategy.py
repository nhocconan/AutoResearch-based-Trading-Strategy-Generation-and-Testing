#!/usr/bin/env python3
"""
12h_TRIX_VolumeSpike_ChopRegime_v1
Hypothesis: TRIX (triple-smoothed EMA) momentum with volume confirmation and choppiness regime filter captures sustained moves while avoiding whipsaws. Long when TRIX rising above zero with volume spike in trending market (CHOP < 38.2); short when TRIX falling below zero with volume spike in trending market. Uses 1w EMA50 for higher timeframe trend alignment. Designed for low trade frequency (target: 50-150 total over 4 years) to minimize fee drag while capturing strong momentum in both bull and bear markets.
"""

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
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1w EMA50 for higher timeframe trend filter
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate TRIX (15-period triple EMA of ROC)
    # TRIX = EMA(EMA(EMA(ROC), 15), 15), 15) * 100
    if len(close) < 15:
        return np.zeros(n)
    roc = np.diff(np.log(close), prepend=np.log(close[0]))  # approximate ROC
    ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = ema3 * 100
    
    # Calculate 1d ATR(14) for choppiness indicator
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1w = df_1d['high'].values
    low_1w = df_1d['low'].values
    close_1w = df_1d['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR,14) / (max(high,14) - min(low,14))) / log10(14)
    max_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    denominator = max_high - min_low
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    chop = 100 * np.log10(pd.Series(atr).rolling(window=14, min_periods=14).sum().values / denominator) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike detection (volume > 2.0x 20-period EMA)
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (volume_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(100, 50, 15, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(trix[i]) or
            np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Higher timeframe trend filter (1w EMA50)
        uptrend_htf = close[i] > ema_50_1w_aligned[i]
        downtrend_htf = close[i] < ema_50_1w_aligned[i]
        
        # Regime filter: trending market (CHOP < 38.2)
        trending_market = chop_aligned[i] < 38.2
        
        # Long logic: TRIX rising above zero with volume spike in uptrend + trending market
        if trix[i] > 0 and trix[i] > trix[i-1] and volume_spike[i] and uptrend_htf and trending_market:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: TRIX falling below zero with volume spike in downtrend + trending market
        elif trix[i] < 0 and trix[i] < trix[i-1] and volume_spike[i] and downtrend_htf and trending_market:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: TRIX crosses zero or trend weakens or market becomes choppy
        elif position == 1 and (trix[i] <= 0 or not uptrend_htf or not trending_market):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (trix[i] >= 0 or not downtrend_htf or not trending_market):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_TRIX_VolumeSpike_ChopRegime_v1"
timeframe = "12h"
leverage = 1.0