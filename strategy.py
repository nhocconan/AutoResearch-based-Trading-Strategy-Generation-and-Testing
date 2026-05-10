#!/usr/bin/env python3
# 4h_TRIX_ZeroCross_VolumeSpike_TrendFilter
# Hypothesis: Uses TRIX (1-period ROC of triple-smoothed EMA) zero cross with volume spike and 1d EMA trend filter.
# TRIX is a momentum oscillator that filters out insignificant price movements and highlights trend changes.
# Combines with volume confirmation to avoid false signals and 1d trend filter to align with higher timeframe bias.
# Designed for low trade frequency (<30/year) to minimize fee drag while capturing momentum shifts in bull/bear markets.

name = "4h_TRIX_ZeroCross_VolumeSpike_TrendFilter"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate TRIX (15-period triple EMA of 1-period ROC)
    # TRIX = EMA(EMA(EMA(ROC), 15), 15), 15) where ROC = (close - close.shift(1)) / close.shift(1)
    roc = np.diff(np.log(close), prepend=np.log(close[0]))  # log returns approximation of ROC
    ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = ema3  # This is the TRIX oscillator
    
    # Get 1d EMA for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume moving average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for TRIX and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(trix[i]) or np.isnan(trix[i-1]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 1d
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 2.0
        
        if position == 0:
            # Long entry: TRIX crosses above zero with volume confirmation and 1d uptrend
            if trix[i] > 0 and trix[i-1] <= 0 and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX crosses below zero with volume confirmation and 1d downtrend
            elif trix[i] < 0 and trix[i-1] >= 0 and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero or trend turns down
            if trix[i] < 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero or trend turns up
            if trix[i] > 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals