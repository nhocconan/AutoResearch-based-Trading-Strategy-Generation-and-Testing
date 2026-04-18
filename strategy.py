#!/usr/bin/env python3
"""
6h_Stochastic_Bollinger_Reversal_v1
Stochastic oscillator with Bollinger Band reversal signals:
- Long when Stoch %K < 20 (oversold) and price touches lower BB, exit when %K > 80
- Short when Stoch %K > 80 (overbought) and price touches upper BB, exit when %K < 20
- Uses 1d trend filter: only long when price > 1d EMA50, short when price < 1d EMA50
- Designed for 12-30 trades/year per symbol
Works in ranging markets (mean reversion) and filters against strong trends
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic oscillator %K and %D."""
    n = len(high)
    lowest_low = np.full(n, np.nan)
    highest_high = np.full(n, np.nan)
    
    for i in range(n):
        if i < k_period - 1:
            continue
        lowest_low[i] = np.min(low[i - k_period + 1:i + 1])
        highest_high[i] = np.max(high[i - k_period + 1:i + 1])
    
    k_percent = np.full(n, np.nan)
    d_percent = np.full(n, np.nan)
    
    for i in range(n):
        if np.isnan(lowest_low[i]) or np.isnan(highest_high[i]) or highest_high[i] == lowest_low[i]:
            continue
        k_percent[i] = 100 * (close[i] - lowest_low[i]) / (highest_high[i] - lowest_low[i])
    
    # Calculate %D as SMA of %K
    k_series = pd.Series(k_percent)
    d_percent = k_series.rolling(window=d_period, min_periods=d_period).mean().values
    
    return k_percent, d_percent

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_series = pd.Series(close)
    ma = close_series.rolling(window=period, min_periods=period).mean().values
    std = close_series.rolling(window=period, min_periods=period).std().values
    upper = ma + std_dev * std
    lower = ma - std_dev * std
    return upper, lower, ma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Bollinger Bands (20, 2)
    upper_bb, lower_bb, middle_bb = calculate_bollinger_bands(close, 20, 2.0)
    
    # Calculate Stochastic (14, 3)
    k_percent, d_percent = calculate_stochastic(high, low, close, 14, 3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need 50 for EMA50 + buffer
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper_bb[i]) or 
            np.isnan(lower_bb[i]) or np.isnan(k_percent[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1d EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: oversold Stoch + price at lower BB + bullish trend filter
            if (k_percent[i] < 20 and 
                close[i] <= lower_bb[i] * 1.005 and  # allow small tolerance
                price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short: overbought Stoch + price at upper BB + bearish trend filter
            elif (k_percent[i] > 80 and 
                  close[i] >= upper_bb[i] * 0.995 and  # allow small tolerance
                  price_below_ema):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Stoch overbought or price touches upper BB
            if k_percent[i] > 80 or close[i] >= upper_bb[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Stoch oversold or price touches lower BB
            if k_percent[i] < 20 or close[i] <= lower_bb[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Stochastic_Bollinger_Reversal_v1"
timeframe = "6h"
leverage = 1.0