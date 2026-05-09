#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Trix_Trend_Volume_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for TRIX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1d data for volume filter (volume spike detection)
    vol_series_1d = pd.Series(df_1d['volume'].values)
    vol_ma_1d = vol_series_1d.rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = df_1d['volume'].values > (vol_ma_1d * 1.5)
    
    # Calculate TRIX (15-period triple EMA)
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix_raw = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix_raw[0] = 0  # First value has no previous
    
    # Align TRIX and volume spike to 4h
    trix_4h = align_htf_to_ltf(prices, df_1d, trix_raw)
    volume_spike_4h = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30  # Need enough data for TRIX calculation
    
    for i in range(start_idx, n):
        if np.isnan(trix_4h[i]) or np.isnan(volume_spike_4h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trix_val = trix_4h[i]
        vol_spike = volume_spike_4h[i]
        
        if position == 0:
            # Enter long: TRIX crosses above zero with volume spike
            if trix_val > 0 and trix_4h[i-1] <= 0 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below zero with volume spike
            elif trix_val < 0 and trix_4h[i-1] >= 0 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below zero
            if trix_val < 0 and trix_4h[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero
            if trix_val > 0 and trix_4h[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals