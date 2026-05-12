#!/usr/bin/env python3
name = "4h_TRIX_VolumeSpike_1dTrend"
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
    
    # === TRIX (15-period EMA applied 3 times) ===
    # TRIX = % change of triple-smoothed EMA
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = ema3.pct_change() * 100  # percentage change
    
    # === 1d EMA34 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 1d Volume spike filter ===
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (2.0 * vol_avg_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Allow time for TRIX calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix.iloc[i]) if hasattr(trix, 'iloc') else np.isnan(trix[i]) or
            np.isnan(ema34_1d_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        trix_val = trix.iloc[i] if hasattr(trix, 'iloc') else trix[i]
        
        if position == 0:
            # Long: TRIX crosses above zero + above daily EMA34 + volume spike
            if (trix_val > 0 and
                close[i] > ema34_1d_aligned[i] and
                vol_spike_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero + below daily EMA34 + volume spike
            elif (trix_val < 0 and
                  close[i] < ema34_1d_aligned[i] and
                  vol_spike_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX crosses below zero or below EMA34
            if trix_val < 0 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX crosses above zero or above EMA34
            if trix_val > 0 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals