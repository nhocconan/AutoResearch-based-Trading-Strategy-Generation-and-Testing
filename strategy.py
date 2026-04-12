#!/usr/bin/env python3
"""
4h_1d_1w_triple_timeframe_momentum
Hypothesis: Combine 4h momentum (RSI), 1d trend (EMA200), and 1w structure (price vs weekly high/low) for high-conviction trades.
Only takes long when: 4h RSI > 55 (bullish momentum), price > 1d EMA200 (uptrend), and price > 1w low + 20% of weekly range (strong above weekly support).
Only takes short when: 4h RSI < 45 (bearish momentum), price < 1d EMA200 (downtrend), and price < 1w high - 20% of weekly range (weak below weekly resistance).
Uses 0.25 position size to limit risk. Designed for 4-8 trades/year per symbol, avoiding fee drag.
"""

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
    
    # Get daily data for trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA200 for trend
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Get weekly data for structure
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly high and low for structure
    weekly_high = high_1w
    weekly_low = low_1w
    
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Calculate 4h RSI for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Calculate weekly range and levels
        weekly_range = weekly_high_aligned[i] - weekly_low_aligned[i]
        if weekly_range <= 0:
            signals[i] = 0.0
            continue
            
        weekly_support = weekly_low_aligned[i] + 0.2 * weekly_range  # 20% above weekly low
        weekly_resistance = weekly_high_aligned[i] - 0.2 * weekly_range  # 20% below weekly high
        
        # Long conditions: bullish momentum + uptrend + above weekly support
        if (rsi[i] > 55 and 
            close[i] > ema200_1d_aligned[i] and 
            close[i] > weekly_support and
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short conditions: bearish momentum + downtrend + below weekly resistance
        elif (rsi[i] < 45 and 
              close[i] < ema200_1d_aligned[i] and 
              close[i] < weekly_resistance and
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions: momentum divergence
        elif position == 1 and rsi[i] < 50:
            position = 0
            signals[i] = 0.0
        elif position == -1 and rsi[i] > 50:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_1w_triple_timeframe_momentum"
timeframe = "4h"
leverage = 1.0