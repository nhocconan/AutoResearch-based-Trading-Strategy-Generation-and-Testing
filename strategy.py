#!/usr/bin/env python3

# 6h_RSI2_MeanReversion_Scalp
# Hypothesis: Extreme RSI(2) levels indicate mean reversion opportunities on 6h timeframe.
# Uses RSI(2) < 10 for long and > 90 for short, with 6h EMA50 trend filter to avoid counter-trend trades.
# Works in both bull and bear markets by aligning with higher timeframe trend while capturing short-term reversals.
# Targets 60-120 total trades over 4 years (15-30/year) with low turnover to minimize fee drag.

name = "6h_RSI2_MeanReversion_Scalp"
timeframe = "6h"
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
    
    # RSI(2) for mean reversion signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 6h EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if np.isnan(rsi[i]) or np.isnan(ema_50[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: RSI(2) extremely oversold + price above EMA50 (uptrend filter)
            if rsi[i] < 10 and close[i] > ema_50[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI(2) extremely overbought + price below EMA50 (downtrend filter)
            elif rsi[i] > 90 and close[i] < ema_50[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI returns to neutral or trend breaks down
            if rsi[i] > 50 or close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI returns to neutral or trend breaks up
            if rsi[i] < 50 or close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals