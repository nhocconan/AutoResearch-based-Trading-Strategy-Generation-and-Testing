#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_ChopRegime
Hypothesis: TRIX (triple-smoothed EMA) crossing zero with volume spike (>2.0x 20-bar avg) and choppiness regime filter (CHOP < 50) captures strong momentum moves while avoiding whipsaws in choppy markets. Uses ATR(14) stoploss (2.0) and discrete sizing (0.25). Designed to work in both bull and bear markets via zero-cross signal and regime filter.
"""

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
    
    # Get 1d data for HTF chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate TRIX on 4h (15-period triple EMA, then ROC)
    # TRIX = 100 * (EMA3(close) - EMA3(close)_prev) / EMA3(close)_prev
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    
    # Calculate ROC of triple EMA: (current - prev) / prev * 100
    trix_raw = np.zeros_like(close)
    trix_raw[:] = np.nan
    trix_raw[1:] = 100 * (ema3[1:] - ema3[:-1]) / ema3[:-1]
    
    # Get 4h data for ATR stoploss
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range for ATR
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_4h = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index regime filter on 1d (CHOP < 50 = strong trending regime)
    n_chop = 14
    tr_1d = []
    for i in range(1, len(high_1d)):
        tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        tr_1d.append(tr)
    tr_1d = np.concatenate([[np.nan], tr_1d])
    
    atr_sum = pd.Series(tr_1d).rolling(window=n_chop, min_periods=n_chop).sum().values
    max_minus_min = pd.Series(high_1d - low_1d).rolling(window=n_chop, min_periods=n_chop).max().values
    chop_raw = 100 * np.log10(atr_sum / (n_chop * max_minus_min)) / np.log10(n_chop)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(30, 15, 20, 14, 14)  # TRIX warmup, vol MA, ATR, Chop
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_raw[i]) or 
            np.isnan(atr_4h_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(chop_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        trix_val = trix_raw[i]
        atr_val = atr_4h_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        chop_val = chop_aligned[i]
        
        # Regime filter: only trade in strong trending markets (CHOP < 50)
        in_strong_trend = chop_val < 50
        
        # Volume spike condition: current volume > 2.0x 20-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: TRIX zero-cross with volume and trend
            # Long: TRIX crosses above zero with volume spike
            long_signal = (trix_val > 0) and (trix_raw[i-1] <= 0) and volume_spike and in_strong_trend
            # Short: TRIX crosses below zero with volume spike
            short_signal = (trix_val < 0) and (trix_raw[i-1] >= 0) and volume_spike and in_strong_trend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Stoploss: price moves against position by 2.0*ATR
            if close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. TRIX cross below zero (exit long)
            elif trix_val < 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Stoploss: price moves against position by 2.0*ATR
            if close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. TRIX cross above zero (exit short)
            elif trix_val > 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0