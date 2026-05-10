#!/usr/bin/env python3
# 1D_KAMA_Trend_Filter_RSI_MeanReversion
# Hypothesis: KAMA trend direction + RSI mean reversion provides robust entry signals across market regimes.
# KAMA adapts to volatility, reducing false signals in choppy markets. RSI identifies overbought/oversold conditions.
# In uptrends, buy RSI < 40 pullbacks; in downtrends, sell RSI > 60 bounces. Works in both bull and bear markets.
# Uses 1h timeframe for entry timing with 1d KAMA trend filter and RSI.
# Targets 20-30 trades per year with strict entry conditions to minimize fee drag.

name = "1D_KAMA_Trend_Filter_RSI_MeanReversion"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) calculation
    # ER (Efficiency Ratio) = |change| / sum(|changes|)
    change = np.abs(np.diff(close, prepend=close[0]))
    direction = np.abs(np.subtract(close, np.roll(close, 1)))
    direction[0] = 0  # First element has no previous
    
    # Avoid division by zero
    er = np.where(change > 0, direction / change, 0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    
    for i in range(1, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = np.zeros(n)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend based on price vs KAMA
        is_uptrend = close[i] > kama[i]
        is_downtrend = close[i] < kama[i]
        
        if position == 0:
            # Long entry: RSI < 40 (oversold) in uptrend + volume confirmation
            if rsi[i] < 40 and is_uptrend and volume[i] > vol_threshold[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: RSI > 60 (overbought) in downtrend + volume confirmation
            elif rsi[i] > 60 and is_downtrend and volume[i] > vol_threshold[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI > 60 (overbought) or trend reversal
            if rsi[i] > 60 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI < 40 (oversold) or trend reversal
            if rsi[i] < 40 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals