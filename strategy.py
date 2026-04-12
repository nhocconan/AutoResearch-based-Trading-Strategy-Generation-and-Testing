#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_cci_reversal_v1
# Uses daily CCI(20) for mean reversion signals and weekly trend filter.
# In range-bound markets: long when CCI < -100, short when CCI > +100.
# Weekly trend filter: only take long signals when price > weekly SMA(50),
# only take short signals when price < weekly SMA(50).
# This avoids fighting the trend and focuses on mean reversion within the trend.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

name = "6h_1d_cci_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for CCI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate CCI(20) on daily data
    tp = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3  # Typical Price
    ma = tp.rolling(window=20, min_periods=20).mean()
    md = tp.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - ma) / (0.015 * md)
    cci_values = cci.values
    
    # Align CCI to 6h timeframe
    cci_aligned = align_htf_to_ltf(prices, df_1d, cci_values)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly SMA(50) for trend filter
    sma_50 = df_1w['close'].rolling(window=50, min_periods=50).mean().values
    sma_aligned = align_htf_to_ltf(prices, df_1w, sma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not ready
        if np.isnan(cci_aligned[i]) or np.isnan(sma_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Mean reversion signals from daily CCI
        long_signal = cci_aligned[i] < -100
        short_signal = cci_aligned[i] > 100
        
        # Trend filter from weekly SMA
        uptrend = close[i] > sma_aligned[i]
        downtrend = close[i] < sma_aligned[i]
        
        # Entry logic: only trade mean reversion in direction of trend
        if long_signal and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit when CCI reverts to mean (between -50 and +50)
        elif -50 <= cci_aligned[i] <= 50 and position != 0:
            position = 0
            signals[i] = 0.0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals