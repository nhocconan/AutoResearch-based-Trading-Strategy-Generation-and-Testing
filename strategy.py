#!/usr/bin/env python3
name = "6h_Aggressive_Force_Index_Breakout_1wTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 300:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1w data for trend ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === 1w EMA34 for trend ===
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === Aggressive Force Index (AFI) on 6m: EMA(13) of (Close - Prev Close) * Volume ===
    # Calculate raw Force Index: (Close - Previous Close) * Volume
    price_change = close - np.roll(close, 1)
    price_change[0] = 0  # First value has no previous close
    fi_raw = price_change * volume
    
    # EMA(13) of Force Index
    afi = pd.Series(fi_raw).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # === Volume spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(300, 34, 20, 13)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(afi[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: AFI crosses above zero + volume spike + 1w trend up
            if (afi[i] > 0 and afi[i-1] <= 0 and 
                volume_spike[i] and
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: AFI crosses below zero + volume spike + 1w trend down
            elif (afi[i] < 0 and afi[i-1] >= 0 and 
                  volume_spike[i] and
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: AFI crosses below zero or trend breaks
            if afi[i] < 0 or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: AFI crosses above zero or trend breaks
            if afi[i] > 0 or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals