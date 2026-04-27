#!/usr/bin/env python3
"""
6h_SuperTrend_Adaptive_ATR_RSI_Exit
Hypothesis: Uses SuperTrend (ATR=10, multiplier=3) on 6h as primary trend filter, with RSI(14) exit on 6h to avoid whipsaws. Designed for medium-term trend following with controlled trade frequency (~15-25 trades/year) to minimize fee drag. Works in bull markets by riding trends and in bear markets by avoiding false breaks during consolidation via RSI exhaustion filters.
"""

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(10)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # SuperTrend calculation
    upper_band = (high + low) / 2 + 3 * atr
    lower_band = (high + low) / 2 - 3 * atr
    
    supertrend = np.full(n, np.nan)
    direction = np.full(n, 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(10, n):
        if close[i] > upper_band[i-1]:
            direction[i] = 1
        elif close[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # RSI(14) for exit signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(supertrend[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above SuperTrend (uptrend)
            if close[i] > supertrend[i]:
                signals[i] = size
                position = 1
            # Short: price below SuperTrend (downtrend)
            elif close[i] < supertrend[i]:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: RSI > 70 (overbought) or price crosses below SuperTrend
            if rsi[i] > 70 or close[i] < supertrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI < 30 (oversold) or price crosses above SuperTrend
            if rsi[i] < 30 or close[i] > supertrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_SuperTrend_Adaptive_ATR_RSI_Exit"
timeframe = "6h"
leverage = 1.0