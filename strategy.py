#!/usr/bin/env python3
"""
12h_WRMS_TrendFollow_VolumeConfirm
Strategy: Wave Root Mean Square trend detection with volume confirmation.
Long: WRMS > threshold + volume > 1.5x average
Short: WRMS < -threshold + volume > 1.5x average
Exit: WRMS crosses zero
Position size: 0.25
Designed to capture trending moves with volume confirmation while avoiding choppy markets.
Timeframe: 12h
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
    
    # Calculate WRMS (Wave Root Mean Square) - detrended price momentum
    # WRMS = sqrt(mean((price - SMA)^2)) * sign(price - SMA)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    detrended = close - sma_20
    squared = detrended ** 2
    mean_squared = pd.Series(squared).rolling(window=10, min_periods=10).mean().values
    wrms = np.sqrt(mean_squared) * np.sign(detrended)
    
    # Volume confirmation (20-period MA)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # SMA20 + WRMS10
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(wrms[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # WRMS thresholds for trend detection
        wrms_threshold = 0.02  # 2% of price level
        
        wrms_above = wrms[i] > wrms_threshold
        wrms_below = wrms[i] < -wrms_threshold
        wrms_cross_zero = wrms[i] * wrms[i-1] <= 0 if i > 0 else False  # crossed zero
        
        if position == 0:
            # Long: WRMS above threshold + volume filter
            if wrms_above and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: WRMS below threshold + volume filter
            elif wrms_below and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: WRMS crosses below zero
            if wrms_cross_zero and wrms[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: WRMS crosses above zero
            if wrms_cross_zero and wrms[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WRMS_TrendFollow_VolumeConfirm"
timeframe = "12h"
leverage = 1.0