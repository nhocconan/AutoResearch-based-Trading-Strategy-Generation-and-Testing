#!/usr/bin/env python3
# 1d_200DMA_Crossover_RSI_Filter
# Hypothesis: Buy when price crosses above 200-day EMA with RSI < 40 (oversold), sell when crosses below 200-day EMA with RSI > 60 (overbought).
# The 200-day EMA acts as a long-term trend filter, while RSI provides mean-reversion entry signals.
# Works in both bull and bear markets: in bull markets, captures dips in uptrend; in bear markets, sells rallies in downtrend.
# Targets 10-25 trades/year on 1d timeframe.

name = "1d_200DMA_Crossover_RSI_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 200-day EMA for trend filter
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate RSI(14)
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
        # Skip if any critical value is NaN
        if np.isnan(ema_200[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price crosses above EMA200 AND RSI < 40 (oversold in uptrend)
            if close[i] > ema_200[i] and close[i-1] <= ema_200[i-1] and rsi[i] < 40:
                signals[i] = 0.25
                position = 1
            # Short: Price crosses below EMA200 AND RSI > 60 (overbought in downtrend)
            elif close[i] < ema_200[i] and close[i-1] >= ema_200[i-1] and rsi[i] > 60:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below EMA200 OR RSI > 70 (overbought)
            if close[i] < ema_200[i] and close[i-1] >= ema_200[i-1] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above EMA200 OR RSI < 30 (oversold)
            if close[i] > ema_200[i] and close[i-1] <= ema_200[i-1] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals