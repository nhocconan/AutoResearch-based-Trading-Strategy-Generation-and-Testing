#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_Regime
Strategy: TRIX momentum with volume spike and volatility regime filter.
Long: TRIX crosses above zero + volume > 1.5x average + ATR(14) < ATR(50) (low volatility)
Short: TRIX crosses below zero + volume > 1.5x average + ATR(14) < ATR(50)
Exit: TRIX crosses back through zero or volatility regime changes
Position size: 0.25
Designed to capture momentum bursts in low-volatility regimes with volume confirmation.
Timeframe: 4h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate TRIX (15-period EMA applied 3 times)
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = 100 * (ema3.pct_change())
    trix_values = trix.fillna(0).values
    
    # Calculate TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix_values).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Volume confirmation (20-period average)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility regime: ATR(14) < ATR(50) indicates low volatility
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    low_vol_regime = atr14 < atr50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(trix_values[i]) or np.isnan(trix_signal[i]) or 
            np.isnan(volume_ma20[i]) or np.isnan(atr14[i]) or np.isnan(atr50[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Volatility regime filter
        vol_filter = low_vol_regime[i]
        
        # TRIX crossover signals
        trix_cross_above = trix_values[i] > trix_signal[i] and trix_values[i-1] <= trix_signal[i-1]
        trix_cross_below = trix_values[i] < trix_signal[i] and trix_values[i-1] >= trix_signal[i-1]
        
        # Entry conditions
        if position == 0:
            # Long: TRIX bullish crossover + volume + low volatility
            if trix_cross_above and volume_filter and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: TRIX bearish crossover + volume + low volatility
            elif trix_cross_below and volume_filter and vol_filter:
                signals[i] = -0.25
                position = -1
        
        # Exit conditions
        elif position == 1:
            # Exit long: TRIX bearish crossover or volatility regime change
            if trix_cross_below or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX bullish crossover or volatility regime change
            if trix_cross_above or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TRIX_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0