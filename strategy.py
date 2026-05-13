#!/usr/bin/env python3
"""
4h_Trix_Momentum_Volume_Filter
Hypothesis: TRIX (12) crossing zero with 1d EMA50 trend and volume spike confirms momentum.
Long when TRIX crosses above zero in uptrend with volume confirmation.
Short when TRIX crosses below zero in downtrend with volume confirmation.
TRIX filters noise, EMA50 ensures trend alignment, volume avoids false signals.
Target: 20-40 trades/year per symbol to minimize fee drag.
"""

name = "4h_Trix_Momentum_Volume_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # TRIX (12): triple EMA of percent change
    # ROC = (close / close.shift(1) - 1) * 100
    roc = np.zeros(n)
    roc[1:] = (close[1:] / close[:-1] - 1.0) * 100.0
    
    # Triple EMA of ROC
    ema1 = pd.Series(roc).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = ema3  # TRIX value
    
    # Zero line cross signals
    trix_prev = np.roll(trix, 1)
    trix_prev[0] = 0
    trix_cross_up = (trix > 0) & (trix_prev <= 0)
    trix_cross_down = (trix < 0) & (trix_prev >= 0)
    
    # 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        if position == 0:
            # LONG: TRIX crosses up, 1d uptrend, volume confirmation
            if trix_cross_up[i] and uptrend_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses down, 1d downtrend, volume confirmation
            elif trix_cross_down[i] and downtrend_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses down or trend reverses
            if trix_cross_down[i] or not uptrend_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses up or trend reverses
            if trix_cross_up[i] or not downtrend_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals