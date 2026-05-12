#!/usr/bin/env python3
# 1D_KAMA_TREND_WITH_RSI_FILTER
# Hypothesis: KAMA adapts to market noise, providing a reliable trend filter in both bull and bear markets.
# In uptrend (price > KAMA), go long when RSI(14) pulls back from overbought (>50 and rising).
# In downtrend (price < KAMA), go short when RSI(14) bounces from oversold (<50 and falling).
# Uses 1d timeframe to minimize trade frequency and fee drag, targeting 10-25 trades/year.
# KAMA's adaptive nature reduces whipsaws during sideways markets, improving survival in chop.

name = "1D_KAMA_TREND_WITH_RSI_FILTER"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA (Kaufman Adaptive Moving Average) parameters
    fast_ema = 2
    slow_ema = 30
    lookback = 10  # ER lookback period
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=lookback))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smooth ER with simple moving average for stability
    er = pd.Series(er).rolling(window=lookback, min_periods=1).mean().values
    # Calculate smoothing constant SC
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[lookback] = close[lookback]  # Initialize
    for i in range(lookback + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) calculation
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad first value
    rsi = np.concatenate([[50], rsi])  # Start with neutral RSI
    
    # RSI momentum (1-bar change)
    rsi_rising = rsi > np.roll(rsi, 1)
    rsi_falling = rsi < np.roll(rsi, 1)
    rsi_rising[0] = False
    rsi_falling[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 14) + 1  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Uptrend (price > KAMA) + RSI > 50 and rising
            if (close[i] > kama[i] and 
                rsi[i] > 50 and 
                rsi_rising[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend (price < KAMA) + RSI < 50 and falling
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and 
                  rsi_falling[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or RSI overbought
            if (close[i] <= kama[i] or 
                rsi[i] >= 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or RSI oversold
            if (close[i] >= kama[i] or 
                rsi[i] <= 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals