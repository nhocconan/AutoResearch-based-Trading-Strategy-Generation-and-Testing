#!/usr/bin/env python3
"""
Hypothesis: 12h TRIX + volume spike + choppiness regime filter.
- Primary timeframe: 12h, HTF: 1d for TRIX calculation and 1w for chop regime.
- TRIX(12) = triple EMA of close, momentum oscillator. Long when TRIX crosses above zero, short when crosses below zero.
- Volume confirmation: current 12h volume > 2.0 * 20-period 12h volume MA.
- Choppiness regime: only trade when CHOP(14) on 1w < 38.2 (trending market) to avoid whipsaws in ranging markets.
- Discrete signal size: 0.25 to minimize fee churn and control drawdown.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Exit: TRIX crosses zero in opposite direction.
- Works in bull via TRIX momentum, in bear via short signals from TRIX negative crossovers.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate TRIX on 1d (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 12:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # TRIX: triple EMA of close, then percent change
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1) * 100
    trix[0] = 0  # first value undefined
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Calculate Choppiness Index on 1w (HTF) for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR
    
    # ATR14
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr14 / (hh14 - ll14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop, additional_delay_bars=0)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # Need TRIX, volume MA, chop
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        in_trending_regime = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long: TRIX crosses above zero AND volume spike AND trending regime
            if i > 0 and trix_aligned[i-1] <= 0 and trix_aligned[i] > 0 and volume_spike[i] and in_trending_regime:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero AND volume spike AND trending regime
            elif i > 0 and trix_aligned[i-1] >= 0 and trix_aligned[i] < 0 and volume_spike[i] and in_trending_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero
            if i > 0 and trix_aligned[i-1] >= 0 and trix_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero
            if i > 0 and trix_aligned[i-1] <= 0 and trix_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_TRIX_VolumeSpike_ChopRegime_v1"
timeframe = "12h"
leverage = 1.0