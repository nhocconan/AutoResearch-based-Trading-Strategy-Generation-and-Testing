#!/usr/bin/env python3
"""
4H_CAMARILLA_R3_S3_BREAKOUT_1D_VOLUME_SPIKE
Hypothesis: Uses Camarilla R3/S3 levels from daily chart with volume spike confirmation.
Only takes breakouts on above-average volume to filter false signals and target
institutional participation. Works in bull/bear by requiring volume confirmation.
Designed for ~25-35 trades/year on 4h to minimize fee drag.
"""
name = "4H_CAMARILLA_R3_S3_BREAKOUT_1D_VOLUME_SPIKE"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Camarilla levels from previous day
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    for i in range(1, n):
        camarilla_r3[i] = close[i-1] + (high[i-1] - low[i-1]) * 1.1 / 2
        camarilla_s3[i] = close[i-1] - (high[i-1] - low[i-1]) * 1.1 / 2
    
    # 1d data for volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = vol_1d / vol_ma_20d  # Current volume / 20-day average
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to have previous day data
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R3 with volume spike
            if (high[i] > camarilla_r3[i] and 
                vol_spike_aligned[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with volume spike
            elif (low[i] < camarilla_s3[i] and 
                  vol_spike_aligned[i] > 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S3 (reversion to mean)
            if close[i] < camarilla_s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R3 (reversion to mean)
            if close[i] > camarilla_r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals