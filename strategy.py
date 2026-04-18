#!/usr/bin/env python3
"""
1d_TRIX_Zero_Cross_With_WeeklyTrend_and_Volume
Hypothesis: Use TRIX(9) zero cross for momentum on daily timeframe, filtered by weekly trend (EMA34) and volume spike (>1.5x 20-day avg). TRIX captures smoothed momentum reversals, weekly trend ensures direction alignment, volume confirms institutional interest. Designed for low trade frequency (<25/year) to minimize fee drag while capturing multi-day momentum shifts in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # TRIX(9) on daily close
    # TRIX = EMA(EMA(EMA(close, 9), 9), 9); then % change
    ema1 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix = pd.Series(ema3).pct_change() * 100
    trix = trix.fillna(0).values
    
    # Weekly EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume spike: >1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 40  # Need TRIX warmup and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(trix[i]) or 
            np.isnan(ema_1w_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        trix_val = trix[i]
        ema_1w_val = ema_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: TRIX crosses above zero with volume spike and above weekly EMA
            if trix_val > 0 and trix[i-1] <= 0 and vol_spike and close[i] > ema_1w_val:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume spike and below weekly EMA
            elif trix_val < 0 and trix[i-1] >= 0 and vol_spike and close[i] < ema_1w_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: TRIX crosses below zero or below weekly EMA
            if trix_val < 0 or close[i] < ema_1w_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: TRIX crosses above zero or above weekly EMA
            if trix_val > 0 or close[i] > ema_1w_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_TRIX_Zero_Cross_With_WeeklyTrend_and_Volume"
timeframe = "1d"
leverage = 1.0