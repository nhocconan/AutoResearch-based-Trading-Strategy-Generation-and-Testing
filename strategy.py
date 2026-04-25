#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_ChopRegime
Hypothesis: TRIX (triple EMA crossover) filters noise and identifies momentum. 
Long when TRIX crosses above zero with volume spike and choppy regime (CHOP>61.8); 
short when TRIX crosses below zero with volume spike and choppy regime. 
Chop regime filter prevents whipsaw in strong trends, focusing on mean-reversion in ranging markets. 
Discrete position sizing (0.25) limits fee drag. Works in both bull and bear markets by adapting to regime.
Target: 20-50 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for chop regime filter (loaded ONCE)
    df_12h = get_htf_data(prices, '12h')
    
    # Chop regime: Choppiness Index on 12h
    def choppiness_index(high_arr, low_arr, close_arr, period=14):
        atr = np.zeros_like(close_arr)
        tr1 = high_arr[1:] - low_arr[1:]
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr[1:] = tr
        atr[0] = atr[1] if len(atr) > 1 else 0
        
        atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
        hh = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        ll = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        range_hl = hh - ll
        
        chop = np.zeros_like(close_arr)
        chop = 100 * np.log10(atr_sum / (range_hl * period)) / np.log10(period)
        chop = np.where(range_hl > 0, chop, 50.0)  # avoid div by zero
        return chop
    
    chop_12h = choppiness_index(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # TRIX: triple EMA of close, then ROC
    def trix(close_arr, period=15):
        ema1 = pd.Series(close_arr).ewm(span=period, adjust=False, min_periods=period).mean().values
        ema2 = pd.Series(ema1).ewm(span=period, adjust=False, min_periods=period).mean().values
        ema3 = pd.Series(ema2).ewm(span=period, adjust=False, min_periods=period).mean().values
        # ROC of triple EMA
        trix_raw = np.zeros_like(close_arr)
        trix_raw[period:] = (ema3[period:] - ema3[:-period]) / ema3[:-period] * 100
        return trix_raw
    
    trix_vals = trix(close)
    trix_prev = np.roll(trix_vals, 1)
    trix_prev[0] = 0
    trix_cross_up = (trix_vals > 0) & (trix_prev <= 0)
    trix_cross_down = (trix_vals < 0) & (trix_prev >= 0)
    
    # Volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need all indicators ready
    start_idx = max(20, 15*3, 14)  # TRIX(15) needs ~45, volume MA 20, chop 14
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix_vals[i]) or np.isnan(trix_prev[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(chop_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Chop regime: choppy market (CHOP > 61.8) for mean reversion
        is_choppy = chop_12h_aligned[i] > 61.8
        
        if position == 0:
            # Long: TRIX cross up + volume spike + choppy regime
            if trix_cross_up[i] and vol_spike[i] and is_choppy:
                signals[i] = 0.25
                position = 1
            # Short: TRIX cross down + volume spike + choppy regime
            elif trix_cross_down[i] and vol_spike[i] and is_choppy:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: TRIX cross down OR chop regime ends (trending market)
            if trix_cross_down[i] or not is_choppy:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: TRIX cross up OR chop regime ends (trending market)
            if trix_cross_up[i] or not is_choppy:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0