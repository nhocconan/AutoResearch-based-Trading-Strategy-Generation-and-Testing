#!/usr/bin/env python3
name = "6h_TrixVolumeSpike_Regime"
timeframe = "6h"
leverage = 1.0

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
    
    # 1d data for TRIX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # TRIX: triple EMA of log returns (15-period)
    log_ret = np.diff(np.log(close_1d), prepend=np.log(close_1d[0]))
    ema1 = pd.Series(log_ret).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix[0] = 0  # first value has no previous
    
    # 1d volume for spike detection (20-period average)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (vol_ma_1d * 1.5)
    
    # Align TRIX and volume spike to 6h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # 6-day EMA for trend filter on 6h close
    ema6_6h = pd.Series(close).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(15, 20, 6)  # Ensure enough data for TRIX, volume MA, and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(vol_spike_aligned[i]) or 
            np.isnan(ema6_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero, price above EMA6, and volume spike
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and 
                close[i] > ema6_6h[i] and vol_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero, price below EMA6, and volume spike
            elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and 
                  close[i] < ema6_6h[i] and vol_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX crosses below zero or price below EMA6
            if trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 or close[i] < ema6_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX crosses above zero or price above EMA6
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 or close[i] > ema6_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals