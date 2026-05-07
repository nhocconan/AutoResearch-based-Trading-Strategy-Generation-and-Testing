#!/usr/bin/env python3
name = "6h_Riverbank_EMA_Crossover_1wTrend_Filter"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA21 for trend filter
    ema_21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # 6h EMA crossover system (fast/slow)
    ema_fast = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_slow = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume filter: current volume > 1.3 * 30-period average
    vol_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 30)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(ema_fast[i]) or 
            np.isnan(ema_slow[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: EMA9 crosses above EMA21 + weekly uptrend + volume
            if ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1] and close[i] > ema_21_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: EMA9 crosses below EMA21 + weekly downtrend + volume
            elif ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1] and close[i] < ema_21_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: EMA crossover in opposite direction
            if position == 1:
                if ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals