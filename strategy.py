#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX with 1-day volume confirmation and volatility filter
# TRIX (Triple Exponential Average) filters out insignificant price movements and shows the rate of change of a triple-smoothed EMA.
# In trending markets: Go long when TRIX crosses above zero, short when crosses below zero.
# Uses 1-day average volume for confirmation to ensure institutional participation.
# Volatility filter (ATR ratio) avoids whipsaw in low volatility environments.
# Target: 20-50 trades/year per symbol (80-200 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # TRIX calculation: triple-smoothed EMA of close, then rate of change
    # First smoothing
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    # Second smoothing
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    # Third smoothing
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # TRIX = 100 * (today's ema3 - yesterday's ema3) / yesterday's ema3
    trix = np.zeros(n)
    trix[1:] = 100 * (ema3[1:] - ema3[:-1]) / ema3[:-1]
    
    # Get 1d data for volume and volatility filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 20-day average volume for filter
    vol_ma_20d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    # Current day's volume relative to 20-day average
    vol_ratio = vol_1d / vol_ma_20d
    
    # ATR(10) for volatility measurement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # ATR ratio: current ATR / 20-period average ATR (volatility filter)
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr / atr_ma
    
    # Align 1d data to 4h timeframe
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or np.isnan(vol_ratio_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current day's volume > 1.5x 20-day average
        volume_filter = vol_ratio_aligned[i] > 1.5
        
        # Volatility filter: avoid extremely low volatility (choppy) conditions
        vol_filter = atr_ratio_aligned[i] > 0.8
        
        # Entry conditions
        if volume_filter and vol_filter:
            # Long when TRIX crosses above zero
            if trix[i] > 0 and (i == start_idx or trix[i-1] <= 0):
                signals[i] = 0.25
                position = 1
            # Short when TRIX crosses below zero
            elif trix[i] < 0 and (i == start_idx or trix[i-1] >= 0):
                signals[i] = -0.25
                position = -1
        
        # Exit conditions: reverse position when TRIX crosses zero in opposite direction
        if position == 1 and trix[i] < 0:
            signals[i] = 0.0
            position = 0
        elif position == -1 and trix[i] > 0:
            signals[i] = 0.0
            position = 0
        # Hold position if no reversal signal
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "4h_TRIX_1dVolumeVolatilityFilter"
timeframe = "4h"
leverage = 1.0