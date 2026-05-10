#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Uses 1-day Camarilla pivot levels for trend direction and price action.
# Enters long when price breaks above R3 level in an uptrend (price > EMA34) with volume confirmation.
# Enters short when price breaks below S3 level in a downtrend (price < EMA34) with volume confirmation.
# Exits on opposite Camarilla level touch (S1 for long, R1 for short) to limit drawdown.
# Designed for low trade frequency (~20-40/year) to avoid fee drag, works in both bull and bear markets.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low), S2 = close - 0.5*(high-low)
    # S1 = close - 0.105*(high-low)
    R3 = close_1d + 1.1 * (high_1d - low_1d)
    S3 = close_1d - 1.1 * (high_1d - low_1d)
    R1 = close_1d + 0.105 * (high_1d - low_1d)
    S1 = close_1d - 0.105 * (high_1d - low_1d)
    
    # Align Camarilla levels to 12h timeframe
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3)
    R1_12h = align_htf_to_ltf(prices, df_1d, R1)
    S1_12h = align_htf_to_ltf(prices, df_1d, S1)
    
    # 1d EMA34 for trend direction
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(R3_12h[i]) or np.isnan(S3_12h[i]) or 
            np.isnan(ema34_12h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 in uptrend (price > EMA34) with volume
            if (close[i] > R3_12h[i] and 
                close[i-1] <= R3_12h[i-1] and 
                close[i] > ema34_12h[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 in downtrend (price < EMA34) with volume
            elif (close[i] < S3_12h[i] and 
                  close[i-1] >= S3_12h[i-1] and 
                  close[i] < ema34_12h[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches S1 level (mean reversion)
            if close[i] <= S1_12h[i] and close[i-1] > S1_12h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches R1 level (mean reversion)
            if close[i] >= R1_12h[i] and close[i-1] < R1_12h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals