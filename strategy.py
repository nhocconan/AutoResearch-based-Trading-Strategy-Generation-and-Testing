#!/usr/bin/env python3
name = "4h_TRIX_13_Signal_9_VolumeSpike_1dTrend"
timeframe = "4h"
leverage = 1.0

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
    
    # === 1d TRIX indicator ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # TRIX(13,9) = EMA(EMA(EMA(close, 13), 13), 13) then ROC of 9
    ema1 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema2 = pd.Series(ema1).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema3 = pd.Series(ema2).ewm(span=13, adjust=False, min_periods=13).mean().values
    trix = 100 * (pd.Series(ema3).pct_change(periods=9).values)
    
    # TRIX signal line = EMA of TRIX, 9
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # TRIX histogram
    trix_hist = trix - trix_signal
    
    # === 1d Volume spike filter ===
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (2.0 * vol_avg_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # Align TRIX histogram and signal
    trix_hist_aligned = align_htf_to_ltf(prices, df_1d, trix_hist)
    trix_signal_aligned = align_htf_to_ltf(prices, df_1d, trix_signal)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_hist_aligned[i]) or 
            np.isnan(trix_signal_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX histogram crosses above zero + volume spike
            if (trix_hist_aligned[i] > 0 and trix_hist_aligned[i-1] <= 0 and
                vol_spike_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: TRIX histogram crosses below zero + volume spike
            elif (trix_hist_aligned[i] < 0 and trix_hist_aligned[i-1] >= 0 and
                  vol_spike_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX histogram crosses below zero
            if trix_hist_aligned[i] < 0 and trix_hist_aligned[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX histogram crosses above zero
            if trix_hist_aligned[i] > 0 and trix_hist_aligned[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals