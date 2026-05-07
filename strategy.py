#!/usr/bin/env python3
"""
4h_TRIX_Trend_Filter_With_Volume
Hypothesis: TRIX momentum on 4h with 1-day trend filter and volume confirmation.
Works in bull/bear via trend filter. Targets 20-30 trades/year to minimize fee drag.
"""

name = "4h_TRIX_Trend_Filter_With_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # TRIX: triple smoothed EMA of price, then ROC
    # EMA1 = EMA(close, 12)
    # EMA2 = EMA(EMA1, 12)
    # EMA3 = EMA(EMA2, 12)
    # TRIX = (EMA3 - EMA3[1]) / EMA3[1] * 100
    
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # Calculate TRIX as percentage change
    trix_raw = np.zeros_like(close)
    trix_raw[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    
    # 1-day trend filter: EMA of daily close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN
        if (np.isnan(trix_raw[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero AND above 1-day EMA with volume confirmation
            if trix_raw[i] > 0 and trix_raw[i-1] <= 0 and close[i] > ema_1d_aligned[i] and volume[i] > vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero AND below 1-day EMA with volume confirmation
            elif trix_raw[i] < 0 and trix_raw[i-1] >= 0 and close[i] < ema_1d_aligned[i] and volume[i] > vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX crosses below zero
            if trix_raw[i] < 0 and trix_raw[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses above zero
            if trix_raw[i] > 0 and trix_raw[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals