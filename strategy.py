#!/usr/bin/env python3
# 4h_TRIX_ZeroCross_Volume_Trend_Filter
# Hypothesis: Uses TRIX (triple EMA) zero-cross for momentum with volume confirmation and 12h trend filter.
# TRIX filters noise and detects sustained momentum. Works in bull/bear by aligning with 12h trend.
# Entry: TRIX crosses zero with volume > 1.5x 20-period average and 12h trend alignment.
# Exit: TRIX crosses zero in opposite direction. Position size 0.25 to limit drawdown.
# Target: 20-40 trades/year to avoid fee drag.

name = "4h_TRIX_ZeroCross_Volume_Trend_Filter"
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
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate TRIX (15-period triple EMA) on close
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15)
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = (ema3.diff() / ema3.shift(1)) * 100  # Percentage change
    
    # Get 12h EMA(50) for trend direction
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate volume average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for TRIX calculation and volume MA
    start_idx = max(45, 20)  # 3*15 for triple EMA + 20 for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(trix.iloc[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 12h
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # TRIX zero-cross signals
        trix_now = trix.iloc[i]
        trix_prev = trix.iloc[i-1]
        
        if position == 0:
            # Long entry: TRIX crosses above zero with volume confirmation and 12h uptrend
            if trix_prev <= 0 and trix_now > 0 and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX crosses below zero with volume confirmation and 12h downtrend
            elif trix_prev >= 0 and trix_now < 0 and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero
            if trix_prev >= 0 and trix_now < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero
            if trix_prev <= 0 and trix_now > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals