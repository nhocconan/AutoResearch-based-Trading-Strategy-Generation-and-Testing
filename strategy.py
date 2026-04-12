#!/usr/bin/env python3
"""
12h_1d_Camarilla_Reversal_With_Trend
Hypothesis: Use daily Camarilla pivot levels (H3/L3) on 12h timeframe for mean-reversion entries, 
filtered by 1-day trend (EMA50) and volume confirmation (>1.5x 30-period average). 
Long near L3 in uptrend, short near H3 in downtrend. Designed for 12h to reduce trade frequency 
and avoid fee drag, targeting 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Reversal_With_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY CAMARILLA PIVOT LEVELS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot and ranges
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: H3, L3 (most significant for reversals)
    h3_1d = close_1d + (range_1d * 1.1 / 4.0)
    l3_1d = close_1d - (range_1d * 1.1 / 4.0)
    
    # Align to 12h timeframe
    h3_12h = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_12h = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # === DAILY TREND FILTER (EMA50) ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or 
            np.isnan(ema50_12h[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: price near L3 in uptrend (price > EMA50) with volume
        near_l3 = low[i] <= l3_12h[i] * 1.005  # within 0.5% of L3
        uptrend = close[i] > ema50_12h[i]
        long_signal = near_l3 and uptrend and (vol_ratio[i] > 1.5)
        
        # Short: price near H3 in downtrend (price < EMA50) with volume
        near_h3 = high[i] >= h3_12h[i] * 0.995  # within 0.5% of H3
        downtrend = close[i] < ema50_12h[i]
        short_signal = near_h3 and downtrend and (vol_ratio[i] > 1.5)
        
        # Exit: opposite signal or trend change
        exit_long = (position == 1) and (close[i] < ema50_12h[i])
        exit_short = (position == -1) and (close[i] > ema50_12h[i])
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals