#!/usr/bin/env python3
"""
12h_1d_WilliamsVixFix_MeanReversion
Hypothesis: Mean reversion on 12h timeframe using Williams Vix Fix (WVF) to detect oversold/overbought conditions.
In bull markets, buy oversold dips; in bear markets, sell overbought rallies.
Uses 1d ATR for regime filter: only trade when ATR is elevated (volatility regime).
Target: 20-40 trades/year (80-160 total over 4 years) to avoid fee drag.
"""

name = "12h_1d_WilliamsVixFix_MeanReversion"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for WVF and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 22:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Williams Vix Fix (WVF) ---
    # WVF = ((Highest Close in n periods - Low) / Highest Close in n periods) * 100
    # We invert it so high values = oversold (like RSI)
    n_wvf = 22
    highest_close = np.full(len(df_1d), np.nan)
    for i in range(n_wvf, len(df_1d)):
        highest_close[i] = np.max(df_1d['close'].values[i-n_wf:i])
    
    # Fix: use proper variable name
    highest_close = np.full(len(df_1d), np.nan)
    for i in range(n_wvf, len(df_1d)):
        highest_close[i] = np.max(df_1d['close'].values[i-n_wvf:i])
    
    lowest_low = df_1d['low'].values
    wvf = np.where(highest_close != 0, ((highest_close - lowest_low) / highest_close) * 100, 0)
    
    # Invert WVF so higher = oversold (like RSI): WVF_inv = 100 - WVF
    wvf_inv = 100 - wvf
    
    # --- 1d ATR for regime filter ---
    atr_period = 14
    high_low = df_1d['high'].values - df_1d['low'].values
    high_close = np.abs(df_1d['high'].values - df_1d['close'].shift(1).values)
    low_close = np.abs(df_1d['low'].values - df_1d['close'].shift(1).values)
    # Handle first element for shift
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = np.full(len(df_1d), np.nan)
    for i in range(atr_period, len(df_1d)):
        atr[i] = np.mean(tr[i-atr_period:i])
    
    # Align to 12h
    wvf_inv_aligned = align_htf_to_ltf(prices, df_1d, wvf_inv)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # ATR regime: only trade when ATR > 20-period median (elevated volatility)
    atr_median = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        atr_median[i] = np.median(atr[i-20:i]) if not np.isnan(atr[i-20:i]).all() else np.nan
    atr_median_aligned = align_htf_to_ltf(prices, df_1d, atr_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max(WVF period, ATR period, ATR median)
    start_idx = max(n_wvf, atr_period, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if np.isnan(wvf_inv_aligned[i]) or np.isnan(atr_aligned[i]) or np.isnan(atr_median_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade when ATR is elevated (vol regime)
        vol_regime = atr_aligned[i] > atr_median_aligned[i]
        
        if position == 0:
            if vol_regime:
                # Long when WVF_inv > 80 (oversold)
                if wvf_inv_aligned[i] > 80:
                    signals[i] = 0.25
                    position = 1
                # Short when WVF_inv < 20 (overbought)
                elif wvf_inv_aligned[i] < 20:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: WVF_inv < 50 (mean reversion) or vol regime ends
                if wvf_inv_aligned[i] < 50 or not vol_regime:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: WVF_inv > 50 (mean reversion) or vol regime ends
                if wvf_inv_aligned[i] > 50 or not vol_regime:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals