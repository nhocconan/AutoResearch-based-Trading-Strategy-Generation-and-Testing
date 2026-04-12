#!/usr/bin/env python3
"""
12h_1d_camarilla_longshort
Uses 1d Camarilla pivot levels for mean reversion entries on 12h timeframe.
Long when price touches L3 support with volume confirmation, short when touches H3 resistance.
Exits when price reaches opposite H3/L3 or returns to pivot.
Includes volatility filter to avoid choppy markets.
Designed for low trade frequency (target: 12-37/year) with high win rate.
Works in both trending and ranging markets by combining pivot levels with volatility filter.
"""

name = "12h_1d_camarilla_longshort"
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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # Pivot = (High + Low + Close) / 3
    pivot = (high_1d + low_1d + close_1d) / 3
    range_ = high_1d - low_1d
    
    # Camarilla levels
    H3 = pivot + (range_ * 1.1 / 2)
    H4 = pivot + (range_ * 1.1)
    L3 = pivot - (range_ * 1.1 / 2)
    L4 = pivot - (range_ * 1.1)
    
    # Align Camarilla levels to 12h
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Volatility filter: avoid trading when ATR is too high (choppy markets)
    # Use 14-period ATR on 12h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    vol_filter = atr < (atr_ma * 1.5)  # Only trade when ATR below 1.5x its MA
    
    # Volume confirmation: volume > 1.2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after enough data for indicators
        # Skip if data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(vol_filter[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price touches or goes below L3 with volume and vol filter
        if low[i] <= L3_aligned[i] and vol_confirm[i] and vol_filter[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short entry: price touches or goes above H3 with volume and vol filter
        elif high[i] >= H3_aligned[i] and vol_confirm[i] and vol_filter[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions
        elif position == 1 and (high[i] >= H3_aligned[i] or close[i] >= pivot_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (low[i] <= L3_aligned[i] or close[i] <= pivot_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals