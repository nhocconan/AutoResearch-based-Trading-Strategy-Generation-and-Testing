#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Moving Average Convergence Divergence (MACD) with weekly trend filter.
# MACD captures momentum shifts; weekly EMA40 filters for primary trend direction.
# Only take long when MACD crosses above signal line AND price above weekly EMA40.
# Only take short when MACD crosses below signal line AND price below weekly EMA40.
# Weekly trend filter prevents counter-trend trades, reducing whipsaws in choppy markets.
# Works in bull markets (trend-following longs) and bear markets (trend-following shorts).
# Target: 7-25 trades per year (30-100 total over 4 years) for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Daily MACD: fast=12, slow=26, signal=9
    # EMA12
    ema12 = np.zeros(n)
    k12 = 2 / (12 + 1)
    ema12[0] = close[0]
    for i in range(1, n):
        ema12[i] = (close[i] - ema12[i-1]) * k12 + ema12[i-1]
    
    # EMA26
    ema26 = np.zeros(n)
    k26 = 2 / (26 + 1)
    ema26[0] = close[0]
    for i in range(1, n):
        ema26[i] = (close[i] - ema26[i-1]) * k26 + ema26[i-1]
    
    # MACD line
    macd_line = ema12 - ema26
    
    # Signal line (EMA of MACD line)
    signal_line = np.zeros(n)
    k9 = 2 / (9 + 1)
    # Find first valid MACD line value to start signal line
    start_idx = 0
    while start_idx < n and np.isnan(macd_line[start_idx]):
        start_idx += 1
    if start_idx < n:
        signal_line[start_idx] = macd_line[start_idx]
        for i in range(start_idx + 1, n):
            signal_line[i] = (macd_line[i] - signal_line[i-1]) * k9 + signal_line[i-1]
    
    # MACD histogram
    macd_hist = macd_line - signal_line
    
    # Weekly EMA40 for trend filter
    close_1w = df_1w['close'].values
    ema40_1w = np.zeros(len(close_1w))
    k40 = 2 / (40 + 1)
    ema40_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema40_1w[i] = (close_1w[i] - ema40_1w[i-1]) * k40 + ema40_1w[i-1]
    
    # Align weekly EMA40 to daily
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    # Start after MACD and signal line are fully warmed up
    start_idx = max(34, 34 + 8)  # EMA26 needs 26 periods, signal line EMA9 needs 9 more after MACD valid
    
    for i in range(start_idx, n):
        # Skip if weekly EMA40 is not ready
        if np.isnan(ema40_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema40 = ema40_1w_aligned[i]
        macd = macd_line[i]
        signal = signal_line[i]
        
        # MACD crossover signals
        macd_bullish_cross = (macd > signal) and (i > start_idx) and (macd_line[i-1] <= signal_line[i-1])
        macd_bearish_cross = (macd < signal) and (i > start_idx) and (macd_line[i-1] >= signal_line[i-1])
        
        if position == 0:
            # Long: MACD bullish crossover + price above weekly EMA40 (uptrend)
            if macd_bullish_cross and (price > ema40):
                position = 1
                signals[i] = position_size
            # Short: MACD bearish crossover + price below weekly EMA40 (downtrend)
            elif macd_bearish_cross and (price < ema40):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: MACD bearish crossover or price crosses below weekly EMA40
            if macd_bearish_cross or (price < ema40):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: MACD bullish crossover or price crosses above weekly EMA40
            if macd_bullish_cross or (price > ema40):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_MACD_Trend_Filter"
timeframe = "1d"
leverage = 1.0