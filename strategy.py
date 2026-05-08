#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_TRIX_Trend_With_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for TRIX, volume, and close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate TRIX (12-period)
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = np.diff(ema3, prepend=ema3[0]) / ema3 * 100
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Volume spike: current 1d volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma20)
    
    # Align TRIX and volume spike to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    trix_signal_aligned = align_htf_to_ltf(prices, df_1d, trix_signal)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(trix_signal_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above signal line with volume spike
            long_cond = (trix_aligned[i] > trix_signal_aligned[i] and 
                        trix_aligned[i-1] <= trix_signal_aligned[i-1] and
                        volume_spike_aligned[i])
            
            # Short: TRIX crosses below signal line with volume spike
            short_cond = (trix_aligned[i] < trix_signal_aligned[i] and 
                         trix_aligned[i-1] >= trix_signal_aligned[i-1] and
                         volume_spike_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below signal line
            if trix_aligned[i] < trix_signal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above signal line
            if trix_aligned[i] > trix_signal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals