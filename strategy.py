#!/usr/bin/env python3
name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "1d"
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
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA100 for trend filter
    ema_100_1w = pd.Series(close_1w).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_100_1w)
    
    # Load daily data for Camarilla pivot levels
    df_1d_piv = get_htf_data(prices, '1d')
    high_1d_piv = df_1d_piv['high'].values
    low_1d_piv = df_1d_piv['low'].values
    close_1d_piv = df_1d_piv['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    pivot = (high_1d_piv + low_1d_piv + close_1d_piv) / 3
    r1 = close_1d_piv + (high_1d_piv - low_1d_piv) * 1.1 / 12
    s1 = close_1d_piv - (high_1d_piv - low_1d_piv) * 1.1 / 12
    
    # Align pivot levels to daily timeframe (use previous day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1d_piv, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d_piv, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d_piv, s1)
    
    # Volume filter: current volume > 2.5x 20-period average (stricter to reduce trades)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_100_1w_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + above 1w EMA100 + volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_100_1w_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + below 1w EMA100 + volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_100_1w_aligned[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 or below 1w EMA100
            if close[i] < s1_aligned[i] or close[i] < ema_100_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R1 or above 1w EMA100
            if close[i] > r1_aligned[i] or close[i] > ema_100_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals