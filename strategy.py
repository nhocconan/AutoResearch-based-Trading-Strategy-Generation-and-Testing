#!/usr/bin/env python3
name = "1d_TRIX_Zero_Cross_1wTrend"
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
    volume = prices['volume'].values
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # TRIX: 15-period EMA applied 3 times, then rate of change
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = 100 * (ema3.pct_change())
    trix_values = trix.values
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 45  # ensure TRIX and other indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(trix_values[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero + above 1w EMA34 + volume filter
            if (trix_values[i] > 0 and 
                trix_values[i-1] <= 0 and  # crossed above zero
                close[i] > ema_34_1w_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero + below 1w EMA34 + volume filter
            elif (trix_values[i] < 0 and 
                  trix_values[i-1] >= 0 and  # crossed below zero
                  close[i] < ema_34_1w_aligned[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX crosses below zero or below 1w EMA34
            if trix_values[i] < 0 or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX crosses above zero or above 1w EMA34
            if trix_values[i] > 0 or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals