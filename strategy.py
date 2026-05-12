#!/usr/bin/env python3
name = "1d_KAMA_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """
    Kaufman's Adaptive Moving Average (KAMA)
    """
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    
    # Efficiency Ratio
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constant
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    
    # KAMA
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
    volume = prices['volume'].values
    
    # Load weekly data once for trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly KAMA for trend filter (10-period)
    close_weekly = df_weekly['close'].values
    kama_weekly = calculate_kama(close_weekly, er_period=10, fast_sc=2, slow_sc=30)
    kama_weekly_aligned = align_htf_to_ltf(prices, df_weekly, kama_weekly)
    
    # Daily RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(kama_weekly_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly KAMA + RSI > 50 + volume filter
            if (close[i] > kama_weekly_aligned[i] and 
                rsi[i] > 50 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly KAMA + RSI < 50 + volume filter
            elif (close[i] < kama_weekly_aligned[i] and 
                  rsi[i] < 50 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below weekly KAMA
            if close[i] < kama_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above weekly KAMA
            if close[i] > kama_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals