#!/usr/bin/env python3
# 4H_TRIX_VOLUME_SPIKE_1D_TREND_FILTER
# Hypothesis: TRIX (Triple Exponential Average) identifies momentum changes with reduced noise.
# In 1d uptrend (EMA50), go long when TRIX crosses above zero with volume spike (>1.5x avg).
# In 1d downtrend (EMA50), go short when TRIX crosses below zero with volume spike.
# Uses volume confirmation to avoid false breakouts and trend filter to avoid counter-trend trades.
# Target: 20-30 trades/year on 4h timeframe.

name = "4H_TRIX_VOLUME_SPIKE_1D_TREND_FILTER"
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
    
    # Daily data for TRIX and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # TRIX calculation (15-period triple EMA)
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1) * 100
    trix[0] = 0  # First value undefined
    
    # EMA50 for trend filter
    ema50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period) for spike detection
    vol_ma20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Align to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    vol_ma20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20)
    
    # Volume spike detection (current volume > 1.5x 20-period average)
    volume_spike = volume > (1.5 * vol_ma20_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 1d uptrend + TRIX crosses above zero + volume spike
            if (close[i] > ema50_aligned[i] and 
                trix_aligned[i] > 0 and 
                trix_aligned[i-1] <= 0 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: 1d downtrend + TRIX crosses below zero + volume spike
            elif (close[i] < ema50_aligned[i] and 
                  trix_aligned[i] < 0 and 
                  trix_aligned[i-1] >= 0 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or TRIX turns negative
            if (close[i] <= ema50_aligned[i] or 
                trix_aligned[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or TRIX turns positive
            if (close[i] >= ema50_aligned[i] or 
                trix_aligned[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals