#!/usr/bin/env python3
name = "6H_Keltner_RSI_Divergence_Trend"
timeframe = "6h"
leverage = 1.0

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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1-day EMA100 for trend filter
    ema100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Align 1-day EMA100 to 6h timeframe
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Keltner Channel (20, 2.0)
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).ewm(span=10, adjust=False, min_periods=10).mean().values
    upper_keltner = ema20 + 2 * atr
    lower_keltner = ema20 - 2 * atr
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if EMA data not ready
        if np.isnan(ema100_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend
        uptrend = close[i] > ema100_1d_aligned[i]
        downtrend = close[i] < ema100_1d_aligned[i]
        
        if position == 0:
            # Enter long: price touches lower Keltner + RSI oversold (<30) + uptrend
            if close[i] <= lower_keltner[i] and rsi[i] < 30 and uptrend:
                signals[i] = 0.25
                position = 1
            # Enter short: price touches upper Keltner + RSI overbought (>70) + downtrend
            elif close[i] >= upper_keltner[i] and rsi[i] > 70 and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses above EMA20 or RSI overbought
            if close[i] > ema20[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses below EMA20 or RSI oversold
            if close[i] < ema20[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals