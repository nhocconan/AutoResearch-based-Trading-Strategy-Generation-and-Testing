#!/usr/bin/env python3
# 12h_KAMA_Trend_RSI20_80_VolumeFilter_V1
# Hypothesis: KAMA adapts to market efficiency, reducing whipsaw in choppy markets. Combined with RSI extremes (20/80) and volume confirmation, it captures strong momentum moves while avoiding false signals in low volatility. Designed for 12h timeframe to limit trade frequency and avoid fee drag. Works in bull markets via momentum continuation and in bear markets via mean reversion from oversold/overbought levels.

name = "12h_KAMA_Trend_RSI20_80_VolumeFilter_V1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA parameters
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    
    # Calculate Efficiency Ratio and KAMA
    change = np.abs(np.diff(close, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
    # Handle first 10 periods
    er = np.full_like(close, np.nan)
    for i in range(10, len(close)):
        price_change = np.abs(close[i] - close[i-10])
        sum_abs_change = np.sum(np.abs(np.diff(close[i-10:i+1])))
        if sum_abs_change > 0:
            er[i] = price_change / sum_abs_change
        else:
            er[i] = 0
    # Smooth ER
    er_smoothed = np.where(np.isnan(er), 0, er)
    sc = (er_smoothed * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start at index 9
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad RSI to match length
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA + RSI oversold (<20) + volume confirmation
            if close[i] > kama[i] and rsi[i] < 20 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA + RSI overbought (>80) + volume confirmation
            elif close[i] < kama[i] and rsi[i] > 80 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below KAMA or RSI overbought (>70)
            if close[i] < kama[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above KAMA or RSI oversold (<30)
            if close[i] > kama[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals