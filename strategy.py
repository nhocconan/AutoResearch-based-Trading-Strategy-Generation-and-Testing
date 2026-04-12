#!/usr/bin/env python3
"""
4h_1d_KAMA_Trend_Follower_v1
Hypothesis: KAMA adapts to market noise - follows price closely in trends, flattens in ranges.
Go long when price crosses above KAMA AND 1d trend is up (price > 1d EMA50).
Go short when price crosses below KAMA AND 1d trend is down (price < 1d EMA50).
Uses KAMA for adaptive trend following and daily EMA for trend filter to reduce whipsaws.
Targets 20-50 total trades over 4 years to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_KAMA_Trend_Follower_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend
    daily_close = df_1d['close'].values
    daily_ema50 = np.full(len(daily_close), np.nan)
    alpha = 2.0 / (50 + 1)
    for i in range(1, len(daily_close)):
        if np.isnan(daily_ema50[i-1]):
            daily_ema50[i] = daily_close[i]
        else:
            daily_ema50[i] = alpha * daily_close[i] + (1 - alpha) * daily_ema50[i-1]
    
    # 4h data for KAMA calculation
    # Calculate Efficiency Ratio and smoothing constants
    lookback = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate price change
    change = np.abs(np.diff(close, n=lookback))
    # Calculate volatility
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    
    # Avoid division by zero
    er = np.zeros_like(change, dtype=float)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    
    # Calculate smoothing constants
    sc = (er * (2.0/(fast_ema+1) - 2.0/(slow_ema+1)) + 2.0/(slow_ema+1)) ** 2
    
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[lookback] = close[lookback]
    
    # Calculate KAMA
    for i in range(lookback + 1, len(close)):
        if not np.isnan(kama[i-1]) and not np.isnan(sc[i-lookback-1]):
            kama[i] = kama[i-1] + sc[i-lookback-1] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align daily EMA50 to 4h timeframe
    daily_ema50_aligned = align_htf_to_ltf(prices, df_1d, daily_ema50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback + 1, n):
        # Skip if any data invalid
        if np.isnan(kama[i]) or np.isnan(daily_ema50_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below daily EMA50
        daily_close_price = df_1d['close'].values
        daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close_price)
        trend_up = daily_close_aligned[i] > daily_ema50_aligned[i]
        
        # KAMA crossover conditions
        long_signal = close[i] > kama[i] and close[i-1] <= kama[i-1] and trend_up
        short_signal = close[i] < kama[i] and close[i-1] >= kama[i-1] and not trend_up
        
        # Signal logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals