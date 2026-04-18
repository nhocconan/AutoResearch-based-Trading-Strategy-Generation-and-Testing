#!/usr/bin/env python3
"""
4h_TRIX_Volume_Spike_Trend_Filter_v1
Strategy: 4h TRIX momentum with volume spike and 1d EMA200 trend filter.
Long when TRIX crosses above zero with volume spike and price above daily EMA200.
Short when TRIX crosses below zero with volume spike and price below daily EMA200.
Exit on opposite TRIX cross or trend change.
Designed for 4h timeframe: ~25-40 trades/year per symbol (100-160 total over 4 years).
Works in bull/bear via trend filter and momentum confirmation.
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily EMA200 for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # TRIX calculation (15-period EMA applied 3 times)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean()
    trix = (ema3 / ema3.shift(1) - 1) * 100
    trix_values = trix.values
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ma_values = vol_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 45  # need enough for TRIX calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(trix_values[i]) or np.isnan(trix_values[i-1]) or 
            np.isnan(vol_ma_values[i]) or np.isnan(ema_200_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        price_above_ema200 = close[i] > ema_200_aligned[i]
        price_below_ema200 = close[i] < ema_200_aligned[i]
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_values[i]
        
        # TRIX cross signals
        trix_cross_up = trix_values[i-1] <= 0 and trix_values[i] > 0
        trix_cross_down = trix_values[i-1] >= 0 and trix_values[i] < 0
        
        if position == 0:
            # Long: TRIX cross up + volume spike + price above daily EMA200
            if trix_cross_up and vol_spike and price_above_ema200:
                signals[i] = 0.25
                position = 1
            # Short: TRIX cross down + volume spike + price below daily EMA200
            elif trix_cross_down and vol_spike and price_below_ema200:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TRIX cross down or trend change (price below EMA200)
            if trix_cross_down or not price_above_ema200:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX cross up or trend change (price above EMA200)
            if trix_cross_up or not price_below_ema200:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TRIX_Volume_Spike_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0