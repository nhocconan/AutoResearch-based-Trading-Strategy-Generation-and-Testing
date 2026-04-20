#!/usr/bin/env python3
# 4h_1d_Trix_ZeroCross_VolumeFilter
# Hypothesis: TRIX zero-line cross with volume confirmation and ADX trend filter works in both bull and bear markets.
# TRIX filters noise, zero-cross indicates momentum shift. Volume ensures conviction. ADX>20 filters ranging markets.
# Target: 20-30 trades per year to avoid fee drag, works across regimes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Trix_ZeroCross_VolumeFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for TRIX and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Calculate 1d TRIX (15-period EMA of EMA of EMA of ROC) ===
    # ROC of close
    roc = np.diff(close_1d, prepend=close_1d[0]) / np.where(close_1d[:-1] == 0, 1, close_1d[:-1])
    roc = np.append(roc[0], roc)  # same length
    # Three-fold EMA smoothing
    ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    
    # === Calculate 1d ADX (14-period) for trend strength ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.append(close_1d[0], close_1d[:-1]))
    tr3 = np.abs(low_1d - np.append(close_1d[0], close_1d[:-1]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1, (plus_di + minus_di))
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align TRIX and ADX to 4h
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 4h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Get values
        close_val = prices['close'].iloc[i]
        trix_val = trix_aligned[i]
        adx_val = adx_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(trix_val) or np.isnan(adx_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero with volume confirmation and ADX > 20 (trending)
            if i > 0:
                prev_trix = trix_aligned[i-1]
                if (prev_trix <= 0 and trix_val > 0 and vol_ratio_val > 2.0 and adx_val > 20):
                    signals[i] = 0.25
                    position = 1
            # Short: TRIX crosses below zero with volume confirmation and ADX > 20 (trending)
            elif i > 0:
                prev_trix = trix_aligned[i-1]
                if (prev_trix >= 0 and trix_val < 0 and vol_ratio_val > 2.0 and adx_val > 20):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: TRIX crosses below zero
            if i > 0:
                prev_trix = trix_aligned[i-1]
                if prev_trix >= 0 and trix_val < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX crosses above zero
            if i > 0:
                prev_trix = trix_aligned[i-1]
                if prev_trix <= 0 and trix_val > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals