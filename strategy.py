#0.5.1
#!/usr/bin/env python3

# 4h_1d_1w_Camarilla_Pivot_Breakout_Volume
# Hypothesis: Use daily/weekly Camarilla pivot levels (S1, S3, R1, R3) with volume confirmation and 1w trend filter.
# Go long when price breaks above R1 with volume spike in 1w uptrend.
# Go short when price breaks below S1 with volume spike in 1w downtrend.
# Exit when price crosses opposite pivot level (S1 for long, R1 for short) or trend reverses.
# Uses daily pivot levels as structure, volume for confirmation, weekly trend as regime filter.
# Target: 20-40 trades/year (80-160 total over 4 years) to avoid fee drag.

name = "4h_1d_1w_Camarilla_Pivot_Breakout_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily Camarilla pivot levels (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    P = (high_1d + low_1d + close_1d) / 3
    R1 = P + (high_1d - low_1d) * 1.1 / 12
    S1 = P - (high_1d - low_1d) * 1.1 / 12
    R3 = P + (high_1d - low_1d) * 1.1 / 4
    S3 = P - (high_1d - low_1d) * 1.1 / 4
    
    # Align daily pivot levels to 4h
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    
    # Weekly trend filter: price above/below weekly SMA50
    close_1w = df_1w['close'].values
    sma_1w = np.full(len(close_1w), np.nan)
    for i in range(50, len(close_1w)):
        sma_1w[i] = np.mean(close_1w[i-50:i])
    sma_1w_4h = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily data (1 bar) + weekly SMA50 + volume MA20
    start_idx = max(1, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or np.isnan(sma_1w_4h[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume spike in 1w uptrend
            if close[i] > R1_4h[i] and volume_spike[i] and close_1w[-1] > sma_1w_4h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike in 1w downtrend
            elif close[i] < S1_4h[i] and volume_spike[i] and close_1w[-1] < sma_1w_4h[i]:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price crosses below S1 OR 1w trend turns down
                if close[i] < S1_4h[i] or close_1w[-1] < sma_1w_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above R1 OR 1w trend turns up
                if close[i] > R1_4h[i] or close_1w[-1] > sma_1w_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals