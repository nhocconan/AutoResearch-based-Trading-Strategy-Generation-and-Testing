#!/usr/bin/env python3
"""
4h_RSI25_75_EMA200_Trend_Filter
Hypothesis: RSI extreme reversals with EMA200 trend filter on 4h timeframe.
Long when RSI < 25 and price > EMA200 (oversold in uptrend).
Short when RSI > 75 and price < EMA200 (overbought in downtrend).
Exit when RSI returns to neutral (40-60 range).
Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend).
Low-frequency signals to avoid overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate EMA200
    ema200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: covers RSI and EMA200
    warmup = 200
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if np.isnan(rsi[i]) or np.isnan(ema200[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry conditions
        if position == 0:
            # Long: RSI < 25 (oversold) and price > EMA200 (uptrend)
            if rsi[i] < 25 and close[i] > ema200[i]:
                signals[i] = 0.25
                position = 1
                continue
            # Short: RSI > 75 (overbought) and price < EMA200 (downtrend)
            elif rsi[i] > 75 and close[i] < ema200[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions
        elif position == 1:
            # Exit long when RSI returns to neutral (40-60)
            if 40 <= rsi[i] <= 60:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when RSI returns to neutral (40-60)
            if 40 <= rsi[i] <= 60:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI25_75_EMA200_Trend_Filter"
timeframe = "4h"
leverage = 1.0