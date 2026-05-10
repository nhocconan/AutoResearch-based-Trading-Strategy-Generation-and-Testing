#!/usr/bin/env python3

name = "12H_Trix_VolumeSpike_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # TRIX on weekly close
    ema1 = pd.Series(df_1w['close']).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix_raw = ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 100
    trix = trix_raw.fillna(0).values
    trix_1w_aligned = align_htf_to_ltf(prices, df_1w, trix)
    
    # Volume spike filter: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for TRIX and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(trix_1w_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above 0 + volume spike
            if trix_1w_aligned[i] > 0 and trix_1w_aligned[i-1] <= 0 and volume[i] > vol_threshold[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below 0 + volume spike
            elif trix_1w_aligned[i] < 0 and trix_1w_aligned[i-1] >= 0 and volume[i] > vol_threshold[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below 0 or volume drops below average
            if trix_1w_aligned[i] < 0 and trix_1w_aligned[i-1] >= 0 or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above 0 or volume drops below average
            if trix_1w_aligned[i] > 0 and trix_1w_aligned[i-1] <= 0 or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals