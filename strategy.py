#!/usr/bin/env python3
"""
Hypothesis: 1-day KAMA direction with RSI filter and weekly trend alignment.
Trades in direction of weekly trend when price aligns with KAMA and RSI shows momentum.
Designed to work in both bull and bear markets by using weekly trend as filter.
Target: 10-25 trades/year per symbol (40-100 total over 4 years) to minimize fee drag.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA(30) for trend
    close_1w = df_1w['close'].values
    ema_30_1w = pd.Series(close_1w).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_30_1w)
    
    # Calculate KAMA on daily data
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close(t) - close(t-10)|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close(t) - close(t-1)| over 10 periods
    # Handle array dimensions
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # using fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # start after 10 periods
    for i in range(10, len(close)):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need KAMA, RSI, and weekly EMA
    start_idx = max(30, 14, 30)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_30_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        kama_val = kama[i]
        rsi_val = rsi[i]
        trend_1w = ema_30_1w_aligned[i]
        
        # Entry conditions: price aligned with KAMA + RSI momentum + weekly trend
        if position == 0:
            # Long: price > KAMA + RSI > 50 + weekly uptrend
            if close[i] > kama_val and rsi_val > 50 and close[i] > trend_1w:
                signals[i] = size
                position = 1
            # Short: price < KAMA + RSI < 50 + weekly downtrend
            elif close[i] < kama_val and rsi_val < 50 and close[i] < trend_1w:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price < KAMA or RSI < 40 or weekly trend turns down
            if close[i] < kama_val or rsi_val < 40 or close[i] < trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price > KAMA or RSI > 60 or weekly trend turns up
            if close[i] > kama_val or rsi_val > 60 or close[i] > trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_RSI_WeeklyTrendFilter"
timeframe = "1d"
leverage = 1.0