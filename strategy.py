#!/usr/bin/env python3
"""
1d_ThreeD_Rebound_v1
Hypothesis: After a 3-day consecutive drop, price often rebounds on the 4th day. 
Buy on close of 3rd consecutive down day if volume is above average and RSI < 30. 
Exit on next day's close or when RSI > 70. 
Sell on 3rd consecutive up day if volume is above average and RSI > 70. 
Exit on next day's close or when RSI < 30. 
Uses daily timeframe to avoid noise and overtrading. 
Designed to work in both bull and bear markets by capturing mean-reversion moves.
"""

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 10:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average (20-day)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Consecutive up/down days
    price_change = np.diff(close, prepend=close[0])
    up_day = price_change > 0
    down_day = price_change < 0
    
    # Count consecutive up days
    cons_up = np.zeros(n)
    for i in range(1, n):
        if up_day[i]:
            cons_up[i] = cons_up[i-1] + 1
        else:
            cons_up[i] = 0
    
    # Count consecutive down days
    cons_down = np.zeros(n)
    for i in range(1, n):
        if down_day[i]:
            cons_down[i] = cons_down[i-1] + 1
        else:
            cons_down[i] = 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    start_idx = 20  # Need volume average and RSI to stabilize
    
    for i in range(start_idx, n):
        vol_confirm = volume[i] > vol_avg[i]
        
        if position == 0:
            # Long: 3rd consecutive down day, volume confirmation, RSI oversold
            if cons_down[i] == 3 and vol_confirm and rsi[i] < 30:
                signals[i] = size
                position = 1
            # Short: 3rd consecutive up day, volume confirmation, RSI overbought
            elif cons_up[i] == 3 and vol_confirm and rsi[i] > 70:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Long exit: next day's close or RSI overbought
            signals[i] = size
            if rsi[i] > 70:  # Exit on overbought
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short exit: next day's close or RSI oversold
            signals[i] = -size
            if rsi[i] < 30:  # Exit on oversold
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_ThreeD_Rebound_v1"
timeframe = "1d"
leverage = 1.0