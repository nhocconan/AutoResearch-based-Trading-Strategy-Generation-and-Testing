#!/usr/bin/env python3
"""
6h_camarilla_pivot_1d_volatility_filter_v1
Hypothesis: Camarilla pivot levels on daily timeframe provide strong support/resistance levels.
We trade reversals at S3/R3 (80% probability) and breakouts at S4/R4 (continuation).
Volatility filter (ATR ratio) ensures we only trade when volatility is expanding (>1.2x average).
Works in both bull and bear markets by adapting to volatility regime.
Target: 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_volatility_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate daily ATR (14-period)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean()
    
    # Calculate daily Camarilla pivot levels
    # Based on previous day's OHLC
    high_prev = df_1d['high'].shift(1)
    low_prev = df_1d['low'].shift(1)
    close_prev = df_1d['close'].shift(1)
    
    pivot = (high_prev + low_prev + close_prev) / 3
    range_prev = high_prev - low_prev
    
    # Camarilla levels
    s1 = close_prev - (range_prev * 1.1 / 12)
    s2 = close_prev - (range_prev * 1.1 / 6)
    s3 = close_prev - (range_prev * 1.1 / 4)
    s4 = close_prev - (range_prev * 1.1 / 2)
    r1 = close_prev + (range_prev * 1.1 / 12)
    r2 = close_prev + (range_prev * 1.1 / 6)
    r3 = close_prev + (range_prev * 1.1 / 4)
    r4 = close_prev + (range_prev * 1.1 / 2)
    
    # Align all daily data to 6h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4.values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4.values)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14.values)
    
    # Volatility filter: current ATR > 1.2 * average ATR (30-day)
    atr_ma = pd.Series(atr_aligned).rolling(window=30, min_periods=30).mean().values
    vol_expanding = atr_aligned > 1.2 * atr_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(atr_ma[i]) or atr_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches R4 (take profit) or reverses below R3
            if close[i] >= r4_aligned[i] or close[i] < r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price reaches S4 (take profit) or reverses above S3
            if close[i] <= s4_aligned[i] or close[i] > s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: reversal at S3 with expanding volatility
            if (close[i] <= s3_aligned[i] and close[i] > s4_aligned[i] and 
                vol_expanding[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: reversal at R3 with expanding volatility
            elif (close[i] >= r3_aligned[i] and close[i] < r4_aligned[i] and 
                  vol_expanding[i]):
                position = -1
                signals[i] = -0.25
            # Long breakout: price breaks above R4 with expanding volatility
            elif (close[i] > r4_aligned[i] and vol_expanding[i]):
                position = 1
                signals[i] = 0.25
            # Short breakout: price breaks below S4 with expanding volatility
            elif (close[i] < s4_aligned[i] and vol_expanding[i]):
                position = -1
                signals[i] = -0.25
    
    return signals