#!/usr/bin/env python3
name = "4h_TrixVolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def trix(arr, period):
    """TRIX: triple exponential moving average rate of change"""
    ema1 = pd.Series(arr).ewm(span=period, adjust=False, min_periods=period).mean()
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
    ema3 = ema2.ewm(span=period, adjust=False, min_periods=period).mean()
    trix_raw = ema3.pct_change() * 100
    return trix_raw.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for TRIX and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate TRIX(9) on 1d close
    trix_1d = trix(close_1d, 9)
    trix_1d_aligned = align_htf_to_ltf(prices, df_1d, trix_1d)
    
    # Calculate volatility regime using 1d ATR
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)))
    tr2 = np.maximum(np.abs(low_1d - np.roll(close_1d, 1)), tr1)
    tr = np.where(np.arange(len(close_1d)) == 0, high_1d - low_1d, tr2)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Normalize TRIX by ATR to create adaptive threshold
    trix_normalized = trix_1d_aligned / (atr_1d_aligned + 1e-10)
    
    # Volume spike (24-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30  # Ensure TRIX and ATR are ready
    
    for i in range(start_idx, n):
        if np.isnan(trix_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above -0.1 (momentum shift up) + volume spike
            if (trix_normalized[i] > -0.1 and 
                trix_normalized[i-1] <= -0.1 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below +0.1 (momentum shift down) + volume spike
            elif (trix_normalized[i] < 0.1 and 
                  trix_normalized[i-1] >= 0.1 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX crosses below +0.1 (momentum fading)
            if trix_normalized[i] < 0.1 and trix_normalized[i-1] >= 0.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX crosses above -0.1 (momentum fading)
            if trix_normalized[i] > -0.1 and trix_normalized[i-1] <= -0.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals