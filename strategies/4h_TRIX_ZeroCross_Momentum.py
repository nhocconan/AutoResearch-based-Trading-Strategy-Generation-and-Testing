#!/usr/bin/env python3
"""
4h_TRIX_ZeroCross_Momentum
Hypothesis: TRIX (1-period rate of change of a triple-smoothed EMA) crossing zero indicates momentum shifts. 
Long when TRIX crosses above zero with volume confirmation and price above 50-period EMA; 
Short when TRIX crosses below zero with volume confirmation and price below 50-period EMA.
Uses 12h EMA50 trend filter to align with higher timeframe direction. Designed for 20-40 trades/year to minimize fee drift.
Works in both bull and bear by capturing momentum reversals with trend and volume filters.
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate TRIX: triple EMA of close, then 1-period ROC
    # TRIX = 100 * (EMA3 - EMA3_prev) / EMA3_prev
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3_prev = np.roll(ema3, 1)
    ema3_prev[0] = np.nan
    trix = 100 * (ema3 - ema3_prev) / ema3_prev
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for TRIX and EMA to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or 
            np.isnan(trix[i-1]) or
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # TRIX zero cross signals
        trix_cross_up = (trix[i-1] <= 0) and (trix[i] > 0)
        trix_cross_down = (trix[i-1] >= 0) and (trix[i] < 0)
        
        # Trend filter from 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Entry conditions with volume confirmation
        long_entry = trix_cross_up and volume_spike[i] and uptrend
        short_entry = trix_cross_down and volume_spike[i] and downtrend
        
        # Exit on opposite signal
        long_exit = trix_cross_down and volume_spike[i]
        short_exit = trix_cross_up and volume_spike[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.30
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.30
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.30  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.30   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_TRIX_ZeroCross_Momentum"
timeframe = "4h"
leverage = 1.0