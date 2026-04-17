#!/usr/bin/env python3
"""
1d_1W_KAMA_Trend_RSI_Entry
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction.
Enter long when price crosses above KAMA with RSI < 50 (avoid overbought), short when price crosses below KAMA with RSI > 50 (avoid oversold).
Weekly trend filter: only trade in direction of weekly KAMA slope to avoid counter-trend trades in chop.
Designed for 10-30 trades per year to minimize fee drag and work in both bull/bear markets via adaptive trend following.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(change) > 1 else np.zeros_like(change)
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === Daily data for KAMA and RSI ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily KAMA for trend
    kama_1d = calculate_kama(close_1d, er_length=10, fast_sc=2, slow_sc=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Daily RSI(14) for entry filter
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === Weekly data for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly KAMA for trend direction
    kama_1w = calculate_kama(close_1w, er_length=10, fast_sc=2, slow_sc=30)
    # Calculate slope: positive if current > previous
    kama_slope_1w = np.diff(kama_1w, prepend=kama_1w[0])
    kama_slope_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_slope_1w)
    
    signals = np.zeros(n)
    
    # Warmup: covers indicators
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(kama_slope_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry conditions
        if position == 0:
            # Long: price crosses above KAMA + RSI < 50 + weekly KAMA rising
            if close[i] > kama_1d_aligned[i] and close[i-1] <= kama_1d_aligned[i-1]:
                if rsi_1d_aligned[i] < 50 and kama_slope_1w_aligned[i] > 0:
                    signals[i] = 0.25
                    position = 1
                    continue
            # Short: price crosses below KAMA + RSI > 50 + weekly KAMA falling
            elif close[i] < kama_1d_aligned[i] and close[i-1] >= kama_1d_aligned[i-1]:
                if rsi_1d_aligned[i] > 50 and kama_slope_1w_aligned[i] < 0:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Exit conditions: reverse signal at opposite cross
        elif position == 1:
            if close[i] < kama_1d_aligned[i] and close[i-1] >= kama_1d_aligned[i-1]:  # cross below
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if close[i] > kama_1d_aligned[i] and close[i-1] <= kama_1d_aligned[i-1]:  # cross above
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1W_KAMA_Trend_RSI_Entry"
timeframe = "1d"
leverage = 1.0