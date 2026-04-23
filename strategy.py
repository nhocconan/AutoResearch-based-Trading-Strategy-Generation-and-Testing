#!/usr/bin/env python3
"""
Hypothesis: 12h Supertrend with 1d ATR regime filter and volume confirmation.
- Supertrend(10,3) captures trend direction with built-in ATR-based stop/reverse logic
- 1d ATR percentile filter: only trade when volatility is elevated (ATR > 60th percentile of 50-period)
- Volume confirmation: volume > 1.3x 20-period average to avoid low-participation whipsaws
- Works in bull markets via trend following and bear markets via volatility-filtered mean reversion
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe to minimize fee drag
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
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Supertrend components on 12h data
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First element has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Supertrend calculation
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            supertrend[i] = max(upper_band[i], supertrend[i-1])
            direction[i] = 1
        else:
            supertrend[i] = min(lower_band[i], supertrend[i-1])
            direction[i] = -1
    
    # 1d ATR for regime filter
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_1d[0] = 0
    tr2_1d[0] = 0
    tr3_1d[0] = 0
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR percentile filter (60th percentile of 50-period)
    atr_ma_1d = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio_1d = atr_1d / atr_ma_1d
    # Calculate percentile rank manually to avoid look-ahead
    atr_percentile = np.zeros(n)
    lookback = 50
    for i in range(lookback, n):
        window = atr_ratio_1d[i-lookback:i]
        if not np.any(np.isnan(window)):
            atr_percentile[i] = (np.sum(window <= atr_ratio_1d[i]) / len(window)) * 100
        else:
            atr_percentile[i] = 50  # Default if not enough data
    
    # Align 1d ATR percentile to 12h timeframe
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, atr_period, 50)  # Need 20 for volume MA, 10 for ATR, 50 for ATR MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_percentile_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility regime: only trade when ATR is elevated (> 60th percentile)
        high_volatility = atr_percentile_aligned[i] > 60
        
        # Volume confirmation
        volume_spike = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Long: Supertrend uptrend + high volatility + volume spike
            if direction[i] == 1 and high_volatility and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Supertrend downtrend + high volatility + volume spike
            elif direction[i] == -1 and high_volatility and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Supertrend downtrend OR low volatility regime
            if direction[i] == -1 or not high_volatility:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Supertrend uptrend OR low volatility regime
            if direction[i] == 1 or not high_volatility:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Supertrend_1dATR_Regime_VolumeSpike"
timeframe = "12h"
leverage = 1.0