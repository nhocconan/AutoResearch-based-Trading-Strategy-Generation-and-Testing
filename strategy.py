#!/usr/bin/env python3
"""
12h_TRIX_ZeroCross_VolumeSpike_1dTrend
Hypothesis: TRIX (triple-smoothed EMA) zero-cross on 12h captures momentum shifts with reduced whipsaw vs single EMA. Combined with 1d EMA34 trend filter and volume confirmation (current volume > 1.5x 20-period average), it filters false signals in choppy markets. Works in bull/bear by following higher-timeframe trend. Low trade frequency (~20-40/year) due to strict TRIX zero-cross + volume + trend confluence.
"""
name = "12h_TRIX_ZeroCross_VolumeSpike_1dTrend"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # TRIX on 12h: triple EMA of ROC
    # TRIX = EMA(EMA(EMA(ROC, 12), 12), 12)
    roc = np.diff(np.log(close), prepend=np.log(close[0]))  # approximate ROC
    ema1 = pd.Series(roc).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = ema3  # TRIX values
    
    # Align 1d trend to 12h
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: TRIX crosses above zero + 1d uptrend + volume
            if trix[i] > 0 and trix[i-1] <= 0 and ema_34_1d_aligned[i] > 0 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: TRIX crosses below zero + 1d downtrend + volume
            elif trix[i] < 0 and trix[i-1] >= 0 and ema_34_1d_aligned[i] < 0 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position != 0:
            # Minimum holding period of 3 bars to reduce trade frequency
            if bars_since_entry < 3:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Exit: TRIX crosses back through zero
            if position == 1:
                if trix[i] < 0:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if trix[i] > 0:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals