#!/usr/bin/env python3
name = "6h_Trix_Signal_With_Volume_Spike_And_Trend_Filter"
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
    
    # === TRIX CALCULATION (15-period EMA of EMA of EMA) ===
    # TRIX is a momentum oscillator that filters out insignificant price movements
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    
    # Calculate TRIX: percentage change in triple-smoothed EMA
    trix = ema3.pct_change() * 100
    trix_values = trix.values
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix_values).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # === 1D TREND FILTER (EMA50) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)  # Strong volume spike
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_values[i]) or np.isnan(trix_signal[i]) or 
            np.isnan(ema50_1d_6h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: TRIX crosses above signal line AND price above 1d EMA50 AND volume spike
            if (trix_values[i] > trix_signal[i] and 
                trix_values[i-1] <= trix_signal[i-1] and  # confirmed crossover
                close[i] > ema50_1d_6h[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below signal line AND price below 1d EMA50 AND volume spike
            elif (trix_values[i] < trix_signal[i] and 
                  trix_values[i-1] >= trix_signal[i-1] and  # confirmed crossover
                  close[i] < ema50_1d_6h[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: TRIX crosses below signal line OR price breaks below 1d EMA50
            if (trix_values[i] < trix_signal[i] and 
                trix_values[i-1] >= trix_signal[i-1]) or \
               close[i] < ema50_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above signal line OR price breaks above 1d EMA50
            if (trix_values[i] > trix_signal[i] and 
                trix_values[i-1] <= trix_signal[i-1]) or \
               close[i] > ema50_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals