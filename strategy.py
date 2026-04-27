#!/usr/bin/env python3
"""
6h_RSI_Momentum_Divergence
Hypothesis: Enter long when price makes lower low but RSI makes higher low (bullish divergence) with volume confirmation; short when price makes higher high but RSI makes lower high (bearish divergence). Uses 6h timeframe to reduce trade frequency and avoid fee drag. Designed to work in both bull and bear markets by capturing reversals at momentum extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Volume filter: require volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 20  # need 20 for volume MA and RSI stability
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Bullish divergence: price lower low, RSI higher low
            if (i >= 2 and low[i] < low[i-1] and low[i-1] < low[i-2] and
                rsi[i] > rsi[i-1] and rsi[i-1] > rsi[i-2] and
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Bearish divergence: price higher high, RSI lower high
            elif (i >= 2 and high[i] > high[i-1] and high[i-1] > high[i-2] and
                  rsi[i] < rsi[i-1] and rsi[i-1] < rsi[i-2] and
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI crosses above 70 (overbought) or below 40 (momentum loss)
            if rsi[i] > 70 or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI crosses below 30 (oversold) or above 60 (momentum loss)
            if rsi[i] < 30 or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_RSI_Momentum_Divergence"
timeframe = "6h"
leverage = 1.0