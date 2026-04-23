#!/usr/bin/env python3
"""
Hypothesis: 12h TRIX(12) zero-line crossover with 1d EMA50 trend filter and volume confirmation.
- TRIX > 0 indicates bullish momentum; TRIX < 0 indicates bearish momentum
- Long: TRIX crosses above zero AND price > 1d EMA50 AND volume > 1.8x 24-period avg
- Short: TRIX crosses below zero AND price < 1d EMA50 AND volume > 1.8x 24-period avg
- Exit: Opposite TRIX crossover OR price crosses 1d EMA50
- Uses 1d HTF for EMA50 trend filter
- TRIX provides smooth momentum signal reducing whipsaw vs MACD
- Designed for low trade frequency (12-37/year) to minimize fee drag
- Works in bull (buy momentum above zero) and bear (sell momentum below zero)
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
    
    # Volume confirmation: > 1.8x 24-period average (24*12h = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Calculate 1d EMA50 for trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate TRIX(12) on 12h close
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - 1 period ago, then % change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix_raw = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix_raw[0] = 0  # First value undefined
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 24, 12*3)  # Need 50 for EMA, 24 for volume MA, 36 for TRIX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(trix_raw[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        # TRIX zero-line crossover signals
        trix_cross_up = trix_raw[i-1] <= 0 and trix_raw[i] > 0  # Cross above zero
        trix_cross_down = trix_raw[i-1] >= 0 and trix_raw[i] < 0  # Cross below zero
        
        if position == 0:
            # Long: TRIX cross up AND price > 1d EMA50 AND volume confirmation
            if trix_cross_up and volume_confirm and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX cross down AND price < 1d EMA50 AND volume confirmation
            elif trix_cross_down and volume_confirm and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX cross down OR price < 1d EMA50 (trend flip)
            if trix_cross_down or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX cross up OR price > 1d EMA50 (trend flip)
            if trix_cross_up or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_TRIX_ZeroCross_1dEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0