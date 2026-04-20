#!/usr/bin/env python3
# 12h_KAMA_Trend_With_1D_Trend_Filter
# Hypothesis: KAMA adapts to market noise, providing reliable trend signals. In trending markets, KAMA follows price closely; in ranging markets, KAMA flattens. Using 1d EMA50 as higher timeframe trend filter ensures we only trade in the direction of the daily trend, reducing whipsaw. KAMA crossover signals combined with 1d trend filter should yield high-quality trades with low frequency, suitable for 12h timeframe to minimize fee drag.

name = "12h_KAMA_Trend_With_1D_Trend_Filter"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate KAMA on 12h data
    # Efficiency Ratio (ER) = |net change| / sum of absolute changes
    # Smoothing Constant (SC) = [ER * (fastest SC - slowest SC) + slowest SC]^2
    # KAMA = previous KAMA + SC * (price - previous KAMA)
    er_period = 10
    fast_sc = 2 / (2 + 1)  # EMA 2
    slow_sc = 2 / (30 + 1)  # EMA 30
    
    # Calculate price changes
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    
    # Calculate ER
    er = np.zeros(n)
    for i in range(er_period, n):
        net_change = abs(close[i] - close[i-er_period])
        total_change = np.sum(change[i-er_period+1:i+1])
        if total_change > 0:
            er[i] = net_change / total_change
        else:
            er[i] = 0
    
    # Calculate SC
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = er_period  # Need enough data for ER calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Determine trend from 1d EMA50
            uptrend = close[i] > ema50_1d_aligned[i]
            downtrend = close[i] < ema50_1d_aligned[i]
            
            # Long: uptrend + price crosses above KAMA
            if uptrend and close[i] > kama[i] and close[i-1] <= kama[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + price crosses below KAMA
            elif downtrend and close[i] < kama[i] and close[i-1] >= kama[i-1]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below KAMA or trend reverses
            if close[i] < kama[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above KAMA or trend reverses
            if close[i] > kama[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals