#!/usr/bin/env python3
"""
4h Triple Moving Average Convergence with Volume Confirmation
Uses EMA(9), EMA(21), SMA(50) convergence on 4h timeframe with volume spike filter.
Designed to capture strong trend moves with low trade frequency by requiring 
multiple moving averages to align in same direction plus volume confirmation.
Works in both bull and bear markets by following the established trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate moving averages on 4h data
    ema_9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    # Volume spike detection (1.5x 10-period average)
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for SMA(50)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_9[i]) or np.isnan(ema_21[i]) or np.isnan(sma_50[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema9_val = ema_9[i]
        ema21_val = ema_21[i]
        sma50_val = sma_50[i]
        
        if position == 0:
            # Long: EMA9 > EMA21 > SMA50 with volume spike
            if (ema9_val > ema21_val > sma50_val and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: EMA9 < EMA21 < SMA50 with volume spike
            elif (ema9_val < ema21_val < sma50_val and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position: maintain until breakdown
            signals[i] = 0.25
            # Exit: EMA9 crosses below EMA21 (trend change)
            if ema9_val < ema21_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position: maintain until breakout
            signals[i] = -0.25
            # Exit: EMA9 crosses above EMA21 (trend change)
            if ema9_val > ema21_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_TripleMA_Convergence_Volume"
timeframe = "4h"
leverage = 1.0