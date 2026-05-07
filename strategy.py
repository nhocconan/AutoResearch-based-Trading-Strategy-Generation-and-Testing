#!/usr/bin/env python3
# 12h_TRIX_VolumeSpike_1wTrend
# Hypothesis: 12-hour TRIX momentum with weekly trend filter and volume spikes.
# Long: TRIX crosses above zero with weekly uptrend (price > EMA50 weekly) and volume spike (>2x 20-period avg).
# Short: TRIX crosses below zero with weekly downtrend (price < EMA50 weekly) and volume spike.
# Exit: TRIX crosses back to zero. Designed for ~15-30 trades/year to avoid fee drag.
# TRIX filters noise and catches sustained momentum; volume confirms strength; weekly trend avoids counter-trend trades.

name = "12h_TRIX_VolumeSpike_1wTrend"
timeframe = "12h"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate TRIX (12-period)
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - 1-period percent change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    pct_change = (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    pct_change[0] = 0  # First value has no prior
    trix = pct_change * 100  # Scale for readability
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike detection: 2.0x average volume (20-period for stability)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(36, 50, 20)  # Ensure we have TRIX (36), weekly EMA50 (50), and volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero, price above weekly EMA50 (uptrend), volume spike
            if (trix[i] > 0 and trix[i-1] <= 0 and 
                close[i] > ema50_1w_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero, price below weekly EMA50 (downtrend), volume spike
            elif (trix[i] < 0 and trix[i-1] >= 0 and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX crosses back to zero (momentum fade)
            if trix[i] < 0 and trix[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses back to zero (momentum fade)
            if trix[i] > 0 and trix[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals