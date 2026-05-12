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
    
    # === 1d TRIX indicator (13,9) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # TRIX: EMA of EMA of EMA, then ROC
    ema1 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean()
    ema2 = ema1.ewm(span=13, adjust=False, min_periods=13).mean()
    ema3 = ema2.ewm(span=13, adjust=False, min_periods=13).mean()
    trix = pd.Series(ema3).pct_change(periods=1) * 100
    trix_signal = trix.ewm(span=9, adjust=False, min_periods=9).mean()
    
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix.values)
    trix_signal_aligned = align_htf_to_ltf(prices, df_1d, trix_signal.values)
    
    # === 1d Trend filter: EMA34 ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 1d Volume spike filter ===
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (2.0 * vol_avg_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_aligned[i]) or 
            np.isnan(trix_signal_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above signal + above daily EMA34 + volume spike
            if (trix_aligned[i] > trix_signal_aligned[i] and
                trix_aligned[i-1] <= trix_signal_aligned[i-1] and
                close[i] > ema34_1d_aligned[i] and
                vol_spike_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below signal + below daily EMA34 + volume spike
            elif (trix_aligned[i] < trix_signal_aligned[i] and
                  trix_aligned[i-1] >= trix_signal_aligned[i-1] and
                  close[i] < ema34_1d_aligned[i] and
                  vol_spike_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX crosses below signal or below daily EMA34
            if (trix_aligned[i] < trix_signal_aligned[i] and trix_aligned[i-1] >= trix_signal_aligned[i-1]) or \
               close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX crosses above signal or above daily EMA34
            if (trix_aligned[i] > trix_signal_aligned[i] and trix_aligned[i-1] <= trix_signal_aligned[i-1]) or \
               close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals