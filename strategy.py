#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_ChopRegime
Hypothesis: TRIX (triple EMA momentum) with volume spike and chop regime filter on 4h. 
Long when TRIX crosses above zero + volume spike + chop < 61.8 (trending). 
Short when TRIX crosses below zero + volume spike + chop < 61.8. 
Uses discrete sizing (0.25) to limit fee drag. Works in bull/bear via regime filter.
Targets 20-40 trades/year per symbol by requiring confluence of momentum, volume, and regime.
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
    
    # Get 1d data for chop regime filter (choppiness index)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate TRIX (15-period triple EMA of ROC) on 4h close
    # TRIX = EMA(EMA(EMA(ROC, 15), 15), 15) where ROC = (close/close_prev - 1) * 100
    roc = np.zeros_like(close)
    roc[1:] = (close[1:] / close[:-1] - 1) * 100
    
    # Triple EMA of ROC
    ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = ema3  # This is the TRIX indicator
    
    # Calculate Choppiness Index on 1d data (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    # ATR14 (smoothed TR)
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr14_sum = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(atr14_sum / (hh14 - ll14)) / log10(14)
    chop = np.zeros_like(close_1d)
    denominator = hh14 - ll14
    valid = (denominator > 0) & (~np.isnan(denominator)) & (~np.isnan(atr14_sum))
    chop[valid] = 100 * np.log10(atr14_sum[valid] / denominator[valid]) / np.log10(14)
    
    # Align chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for TRIX (15*3=45), chop (20), volume MA (20)
    start_idx = max(45, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix[i]) or np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: trending market (chop < 61.8)
        trending_regime = chop_aligned[i] < 61.8
        
        # TRIX cross signals
        trix_cross_up = (i > 0) and (trix[i] > 0) and (trix[i-1] <= 0)
        trix_cross_down = (i > 0) and (trix[i] < 0) and (trix[i-1] >= 0)
        
        if position == 0:
            # Long setup: TRIX crosses above zero + volume spike + trending regime
            long_setup = trix_cross_up and volume_confirm[i] and trending_regime
            
            # Short setup: TRIX crosses below zero + volume spike + trending regime
            short_setup = trix_cross_down and volume_confirm[i] and trending_regime
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: TRIX crosses below zero OR chop > 61.8 (range regime) OR volume drops
            if (trix[i] < 0) or (chop_aligned[i] >= 61.8) or (not volume_confirm[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: TRIX crosses above zero OR chop > 61.8 (range regime) OR volume drops
            if (trix[i] > 0) or (chop_aligned[i] >= 61.8) or (not volume_confirm[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0