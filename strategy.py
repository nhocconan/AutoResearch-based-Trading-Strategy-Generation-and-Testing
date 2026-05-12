#!/usr/bin/env python3
name = "1d_TRIX_VolumeSpike_Trend"
timeframe = "1d"
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
    
    # Load weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # TRIX (15-period EMA of EMA of EMA) on daily close
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = (ema3.diff() / ema3.shift(1)) * 100  # percentage rate of change
    trix_values = trix.values
    
    # Weekly EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike: current volume > 2.5x 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_values[i]) or np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero + weekly trend up + volume spike
            if (trix_values[i] > 0 and trix_values[i-1] <= 0 and 
                close[i] > ema34_1w_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero + weekly trend down + volume spike
            elif (trix_values[i] < 0 and trix_values[i-1] >= 0 and 
                  close[i] < ema34_1w_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX crosses below zero
            if trix_values[i] < 0 and trix_values[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX crosses above zero
            if trix_values[i] > 0 and trix_values[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals