#!/usr/bin/env python3
"""
1d_WMABreakout_WeeklyTrend_Filter
Hypothesis: On daily timeframe, buy when price breaks above Wilder Moving Average (WMA) with weekly uptrend filter,
sell when price breaks below WMA with weekly downtrend filter. WMA adapts faster than SMA but slower than EMA,
providing good trend-following in both bull and bear markets. Weekly trend filter prevents counter-trend trades.
Target: 15-25 trades/year on 1d timeframe with disciplined entry conditions.
"""

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
    
    # Wilder Moving Average (WMA) - 34 period
    # WMA is similar to Wilder's smoothing used in RSI/ATR
    wma = np.full(n, np.nan)
    alpha = 1 / 34  # Wilder's smoothing constant
    for i in range(34, n):
        if i == 34:
            wma[i] = np.mean(close[0:35])  # Initialize with simple average
        else:
            wma[i] = alpha * close[i] + (1 - alpha) * wma[i-1]
    
    # Weekly trend filter: Weekly WMA 34
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    wma_1w = np.full(len(close_1w), np.nan)
    alpha_1w = 1 / 34
    for i in range(34, len(close_1w)):
        if i == 34:
            wma_1w[i] = np.mean(close_1w[0:35])
        else:
            wma_1w[i] = alpha_1w * close_1w[i] + (1 - alpha_1w) * wma_1w[i-1]
    wma_1w_aligned = align_htf_to_ltf(prices, df_1w, wma_1w)
    
    # Volume confirmation: current volume > 1.5 x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(wma[i]) or np.isnan(wma_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above WMA with weekly uptrend and volume confirmation
            if (close[i] > wma[i] and close[i-1] <= wma[i-1] and  # upward cross
                close[i] > wma_1w_aligned[i] and                  # weekly uptrend
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below WMA with weekly downtrend and volume confirmation
            elif (close[i] < wma[i] and close[i-1] >= wma[i-1] and  # downward cross
                  close[i] < wma_1w_aligned[i] and                  # weekly downtrend
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below WMA or weekly trend turns down
            if (close[i] < wma[i] and close[i-1] >= wma[i-1]) or \
               (close[i] < wma_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above WMA or weekly trend turns up
            if (close[i] > wma[i] and close[i-1] <= wma[i-1]) or \
               (close[i] > wma_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WMABreakout_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0