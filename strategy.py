#!/usr/bin/env python3
"""
4h_TRIX_Volume_Spike_Chop_Regime
Hypothesis: TRIX (triple exponential average) with period 9 detects momentum shifts.
A volume spike (1.5x 20-period average) confirms the move. Choppiness index > 61.8 filters
for ranging markets where TRIX is less reliable. Works in bull/bear by capturing momentum
in trending regimes and avoiding whipsaws in ranges. Target: 20-40 trades/year (80-160 total).
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
    
    # TRIX calculation: triple EMA of log returns
    log_returns = np.diff(np.log(close), prepend=np.log(close[0]))
    ema1 = pd.Series(log_returns).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix[0] = 0  # first value undefined
    
    # 12h data for Choppiness Index
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR14) / (max(HH14) - min(LL14))) / log10(14)
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = np.abs(df_12h['high'] - df_12h['close'].shift(1))
    tr3 = np.abs(df_12h['low'] - df_12h['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hh14 = df_12h['high'].rolling(window=14, min_periods=14).max().values
    ll14 = df_12h['low'].rolling(window=14, min_periods=14).min().values
    chop = 100 * (np.log10(np.nansum(pd.Series(atr14).rolling(14, min_periods=14).sum(), axis=0)) / 
                  np.log10(14)) / np.log10((hh14 - ll14).replace(0, np.nan))
    chop = np.where((hh14 - ll14) == 0, 100, chop)
    chop_4h = align_htf_to_ltf(prices, df_12h, chop, additional_delay_bars=0)
    
    # Volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 14)  # warmup for volume and chop
    
    for i in range(start_idx, n):
        if (np.isnan(trix[i]) or np.isnan(chop_4h[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX turns up with volume spike in trending market (CHOP <= 61.8)
            if trix[i] > trix[i-1] and volume_spike[i] and chop_4h[i] <= 61.8:
                signals[i] = 0.25
                position = 1
            # Short: TRIX turns down with volume spike in trending market
            elif trix[i] < trix[i-1] and volume_spike[i] and chop_4h[i] <= 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: TRIX turns down or chop increases (range)
            if trix[i] < trix[i-1] or chop_4h[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: TRIX turns up or chop increases
            if trix[i] > trix[i-1] or chop_4h[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TRIX_Volume_Spike_Chop_Regime"
timeframe = "4h"
leverage = 1.0