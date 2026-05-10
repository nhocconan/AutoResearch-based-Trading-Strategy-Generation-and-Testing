#!/usr/bin/env python3
# 12h_KAMA_With_1wTrend
# Hypothesis: KAMA adapts to market noise, providing reliable trend signals. Combined with weekly trend filter (price above/below weekly KAMA) and volume confirmation, it captures strong trends while avoiding chop. Weekly trend ensures we only trade in the dominant long-term direction, reducing false signals in ranging markets. Designed for 12h timeframe to target 12-37 trades/year, minimizing fee drag.

name = "12h_KAMA_With_1wTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly KAMA for trend filter
    def calculate_kama(close_vals, er_len=10, fast=2, slow=30):
        n = len(close_vals)
        if n < er_len:
            return np.full(n, np.nan)
        change = np.abs(np.diff(close_vals, n=er_len))
        volatility = np.sum(np.abs(np.diff(close_vals)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros(n)
        kama[0] = close_vals[0]
        for i in range(1, n):
            kama[i] = kama[i-1] + sc[i] * (close_vals[i] - kama[i-1])
        return kama
    
    kama_1w = calculate_kama(df_1w['close'].values)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate 12h KAMA for entry signal
    kama_12h = calculate_kama(close)
    
    # Volume confirmation (24-period MA on 12h = ~12 days)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly KAMA (30), 12h KAMA (30), volume MA (24)
    start_idx = max(30, 24)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_1w_aligned[i]) or 
            np.isnan(kama_12h[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter: price above/below weekly KAMA
        weekly_uptrend = close[i] > kama_1w_aligned[i]
        weekly_downtrend = close[i] < kama_1w_aligned[i]
        
        # 12h KAMA crossover for entry
        kama_cross_up = close[i] > kama_12h[i] and close[i-1] <= kama_12h[i-1]
        kama_cross_down = close[i] < kama_12h[i] and close[i-1] >= kama_12h[i-1]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: weekly uptrend + price crosses above 12h KAMA + volume
            if weekly_uptrend and kama_cross_up and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: weekly downtrend + price crosses below 12h KAMA + volume
            elif weekly_downtrend and kama_cross_down and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly trend turns down OR price crosses below 12h KAMA
            if not weekly_uptrend or (close[i] < kama_12h[i] and close[i-1] >= kama_12h[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: weekly trend turns up OR price crosses above 12h KAMA
            if not weekly_downtrend or (close[i] > kama_12h[i] and close[i-1] <= kama_12h[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals