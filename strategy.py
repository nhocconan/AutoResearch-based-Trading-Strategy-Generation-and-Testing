#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI_Scalper
Hypothesis: On 4h timeframe, use KAMA (Kaufman Adaptive Moving Average) as a dynamic trend filter and RSI for mean-reversion entries. Go long when price is above KAMA and RSI crosses above 30 from below, short when price is below KAMA and RSI crosses below 70 from above. Exit on opposite RSI cross or trend reversal. Designed for 4h to capture trend-following with mean-reversion entries in both bull and bear markets, targeting 20-50 trades/year with low churn.
"""

name = "4h_KAMA_Trend_RSI_Scalper"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) - trend filter
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[t] - close[t-1]| over 10 periods
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # start at index 9
    for i in range(10, n):
        if np.isnan(kama[i-1]) or np.isnan(sc[i]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # First average gain/loss
    avg_gain = np.concatenate([np.full(14, np.nan), [np.mean(gain[1:15])] if len(gain) >= 15 else [np.nan]])
    avg_loss = np.concatenate([np.full(14, np.nan), [np.mean(loss[1:15])] if len(loss) >= 15 else [np.nan]])
    # Smooth subsequent values
    for i in range(15, n):
        if np.isnan(avg_gain[i-1]) or np.isnan(avg_loss[i-1]):
            avg_gain[i] = np.nan
            avg_loss[i] = np.nan
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(rsi[i-1]) if i > 0 else False):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        if position == 0:
            # LONG: Price above KAMA (uptrend) and RSI crosses above 30 from below
            if close[i] > kama[i] and rsi[i-1] < 30 and rsi[i] >= 30:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA (downtrend) and RSI crosses below 70 from above
            elif close[i] < kama[i] and rsi[i-1] > 70 and rsi[i] <= 70:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI crosses below 70 from above or trend reverses (price < KAMA)
            if rsi[i-1] > 70 and rsi[i] <= 70 or close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI crosses above 30 from below or trend reverses (price > KAMA)
            if rsi[i-1] < 30 and rsi[i] >= 30 or close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals