#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_trix_volume_momentum_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d TRIX (15-period EMA of EMA of EMA of close, then ROC)
    close_1d = df_1d['close'].values
    
    # Triple EMA
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean()
    
    # TRIX = 100 * (EMA3 - EMA3.shift(1)) / EMA3.shift(1)
    trix_raw = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix = trix_raw.fillna(0).values
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Align TRIX and signal line to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    trix_signal_aligned = align_htf_to_ltf(prices, df_1d, trix_signal)
    
    # Calculate 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 30 to ensure sufficient data for TRIX calculation
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(trix_aligned[i]) or np.isnan(trix_signal_aligned[i]) or 
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current 12h volume
        vol_current = volume[i]
        vol_surge = vol_current > 1.5 * vol_avg_aligned[i]  # 50% above average
        
        # Long signal: TRIX crosses above signal line with volume surge
        long_signal = (trix_aligned[i] > trix_signal_aligned[i] and 
                      trix_aligned[i-1] <= trix_signal_aligned[i-1] and vol_surge)
        # Short signal: TRIX crosses below signal line with volume surge
        short_signal = (trix_aligned[i] < trix_signal_aligned[i] and 
                       trix_aligned[i-1] >= trix_signal_aligned[i-1] and vol_surge)
        
        # Exit when TRIX crosses back in opposite direction
        exit_long = (trix_aligned[i] < trix_signal_aligned[i] and 
                    trix_aligned[i-1] >= trix_signal_aligned[i-1])
        exit_short = (trix_aligned[i] > trix_signal_aligned[i] and 
                     trix_aligned[i-1] <= trix_signal_aligned[i-1])
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals