#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_ChopRegime
Hypothesis: 4h TRIX (triple EMA) crossover + volume spike (2.5x average) + chop regime filter (CHOP<38.2 for trending). 
Long when TRIX crosses above zero in trending markets, short when crosses below zero. 
Volume confirmation filters false signals. ATR(14) stoploss (2.0x). Discrete sizing 0.30.
Works in both bull/bear via regime filter that only allows entries in trending conditions (low chop).
Timeframe: 4h, uses 1d HTF for chop regime calculation.
Target: 100-180 total trades over 4 years = 25-45/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for chop regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d OHLC for Chop regime calculation ===
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range for 1d
    tr1 = pd.Series(df_1d_high - df_1d_low)
    tr2 = pd.Series(np.abs(df_1d_high - np.roll(df_1d_close, 1)))
    tr3 = pd.Series(np.abs(df_1d_low - np.roll(df_1d_close, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Sum of absolute price changes for Chop denominator
    abs_changes = pd.Series(np.abs(np.diff(df_1d_close, prepend=df_1d_close[0]))).rolling(window=14, min_periods=14).sum().values
    
    # Chop = 100 * log10(sum(TR)/abs_changes) / log10(14)
    chop_1d = 100 * (np.log10(tr_1d.rolling(window=14, min_periods=14).sum().values) - np.log10(abs_changes)) / np.log10(14)
    
    # Align Chop to 4h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 4h TRIX calculation ===
    close = prices['close'].values
    # TRIX: triple EMA, then percent change
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = 100 * (pd.Series(ema3).pct_change(periods=1)).values
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    tr = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))))
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(chop_1d_aligned[i]) or np.isnan(trix[i]) or np.isnan(vol_ma[i]) 
            or np.isnan(atr[i]) or np.isnan(trix[i-1])):  # need previous TRIX for crossover
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        chop = chop_1d_aligned[i]
        trix_now = trix[i]
        trix_prev = trix[i-1]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 2.5x average (strict filter)
        volume_confirmed = volume_now > 2.5 * vol_avg
        
        # Chop regime: only allow entries when market is trending (CHOP < 38.2)
        trending_regime = chop < 38.2
        
        # TRIX crossover signals
        trix_cross_up = (trix_prev <= 0) and (trix_now > 0)
        trix_cross_down = (trix_prev >= 0) and (trix_now < 0)
        
        if position == 0:
            # Enter long: TRIX crosses above zero in trending market with volume confirmation
            if trix_cross_up and trending_regime and volume_confirmed:
                signals[i] = 0.30
                position = 1
                entry_price = price
            # Enter short: TRIX crosses below zero in trending market with volume confirmation
            elif trix_cross_down and trending_regime and volume_confirmed:
                signals[i] = -0.30
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: TRIX crosses below zero
            elif trix_now < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: TRIX crosses above zero
            elif trix_now > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0