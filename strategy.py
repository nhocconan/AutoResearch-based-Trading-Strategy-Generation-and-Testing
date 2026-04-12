#!/usr/bin/env python3
"""
1d_1w_KAMA_Trend_RSI_Momentum_v1
Hypothesis: On 1d timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction and RSI for momentum confirmation. Enter long when KAMA turns up and RSI > 50, enter short when KAMA turns down and RSI < 50. Use 1w timeframe to filter trades: only take long when price > 1w KAMA, short when price < 1w KAMA. Designed for low trade frequency (<25/year) with trend-following logic that works in both bull and bear markets by adapting to volatility.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_KAMA_Trend_RSI_Momentum_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    
    # KAMA calculation function
    def kama(close, er_len=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(change.shape) > 0 else np.sum(np.abs(np.diff(close)))
        # For simplicity, we'll compute volatility as rolling sum of absolute changes
        volatility = pd.Series(change).rolling(window=er_len, min_periods=1).sum().values
        # Avoid division by zero
        er = np.where(volatility != 0, np.abs(np.diff(close, prepend=close[0])) / volatility, 0)
        # Handle first element
        er[0] = 0
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Calculate KAMA on 1d
    kama_1d = kama(close, er_len=10, fast=2, slow=30)
    
    # Calculate KAMA on 1w for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    kama_1w = kama(close_1w, er_len=10, fast=2, slow=30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # RSI calculation
    def rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
        avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        # Handle first period values
        rsi[:period] = 50
        return rsi
    
    rsi_14 = rsi(close, period=14)
    
    # KAMA direction: 1 if rising, -1 if falling, 0 if flat
    kama_dir = np.zeros_like(kama_1d)
    kama_dir[1:] = np.where(kama_1d[1:] > kama_1d[:-1], 1, np.where(kama_1d[1:] < kama_1d[:-1], -1, 0))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_1d[i]) or np.isnan(kama_1w_aligned[i]) or 
            np.isnan(rsi_14[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions
        kama_up = kama_dir[i] == 1
        kama_down = kama_dir[i] == -1
        rsi_bull = rsi_14[i] > 50
        rsi_bear = rsi_14[i] < 50
        price_above_1wkama = close[i] > kama_1w_aligned[i]
        price_below_1wkama = close[i] < kama_1w_aligned[i]
        
        long_entry = kama_up and rsi_bull and price_above_1wkama
        short_entry = kama_down and rsi_bear and price_below_1wkama
        
        # Exit conditions: opposite KAMA turn
        long_exit = kama_down
        short_exit = kama_up
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals