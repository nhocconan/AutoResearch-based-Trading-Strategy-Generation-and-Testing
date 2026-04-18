#!/usr/bin/env python3
"""
4h_RSI2_Trend_Refined
4h strategy using ultra-short RSI(2) for mean reversion with trend filter.
- Long: RSI(2) < 10 + price > EMA(50) (trend filter)
- Short: RSI(2) > 90 + price < EMA(50)
- Exit: RSI(2) crosses back above 50 (long) or below 50 (short)
Designed for ~20-30 trades/year per symbol (80-120 total over 4 years)
Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend)
"""

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # RSI(2) - ultra short for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/period)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # EMA(50) for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA50
    
    for i in range(start_idx, n):
        # Skip if RSI not ready (first values unstable)
        if np.isnan(rsi[i]) or np.isnan(ema_50[i]):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi[i]
        price = close[i]
        ema_val = ema_50[i]
        
        if position == 0:
            # Long: oversold in uptrend
            if rsi_val < 10 and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: overbought in downtrend
            elif rsi_val > 90 and price < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses above 50 (mean reversion complete)
            if rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI crosses below 50 (mean reversion complete)
            if rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI2_Trend_Refined"
timeframe = "4h"
leverage = 1.0