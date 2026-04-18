#!/usr/bin/env python3
"""
4h_4H_200EMA_RSI14_Range_Bounce
Mean reversion strategy on 4h timeframe using 200 EMA as dynamic support/resistance and RSI for entry timing:
- Long when price touches 200 EMA from below and RSI < 30 (oversold)
- Short when price touches 200 EMA from above and RSI > 70 (overbought)
- Exit when price crosses 200 EMA in opposite direction or RSI reaches opposite extreme
- Works in both bull and bear markets by using 200 EMA as dynamic trend filter
"""

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 200 EMA for trend filter
    close_series = pd.Series(close)
    ema_200 = close_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_ma = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    loss_ma = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    rs = gain_ma / (loss_ma + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if EMA not ready
        if np.isnan(ema_200[i]) or np.isnan(rsi[i]):
            continue
        
        if position == 0:
            # Long: price touches EMA from below and RSI oversold
            if low[i] <= ema_200[i] and close[i] > ema_200[i] and rsi[i] < 30:
                signals[i] = 0.25
                position = 1
            # Short: price touches EMA from above and RSI overbought
            elif high[i] >= ema_200[i] and close[i] < ema_200[i] and rsi[i] > 70:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below EMA or RSI overbought
            if close[i] < ema_200[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA or RSI oversold
            if close[i] > ema_200[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_4H_200EMA_RSI14_Range_Bounce"
timeframe = "4h"
leverage = 1.0