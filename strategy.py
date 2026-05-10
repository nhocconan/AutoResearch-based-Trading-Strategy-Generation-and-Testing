#!/usr/bin/env python3
# 4h_TRIX_Trend_Volume_Spike
# Hypothesis: TRIX (triple-smoothed EMA) identifies trend direction with less whipsaw.
# Volume spikes confirm breakout strength. Designed for low trade frequency (<20/year)
# to minimize fee drag. Works in bull markets via trend continuation and in bear via
# mean-reversion at extremes when TRIX flips.

name = "4h_TRIX_Trend_Volume_Spike"
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
    
    # TRIX calculation (15-period triple EMA)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = np.where(ema1[:-30] != 0, (ema3[30:] - ema1[:-30]) / ema1[:-30] * 100, 0)
    # Pad to match length
    trix_padded = np.full(n, np.nan)
    trix_padded[30:] = trix
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.5 * vol_ma  # Require 2.5x average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for TRIX
    
    for i in range(start_idx, n):
        if np.isnan(trix_padded[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX turns positive with volume spike
            if trix_padded[i] > 0 and trix_padded[i-1] <= 0 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX turns negative with volume spike
            elif trix_padded[i] < 0 and trix_padded[i-1] >= 0 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX turns negative
            if trix_padded[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX turns positive
            if trix_padded[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals