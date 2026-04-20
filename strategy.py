#!/usr/bin/env python3
# 12h_KAMA_Direction_RSI20_80_VolumeFilter_V1
# Hypothesis: KAMA adapts to market noise, providing a smooth trend direction signal.
# RSI extremes (<20 for oversold, >80 for overbought) with volume confirmation capture
# high-probability reversals in both bull and bear markets. Designed for 12h to minimize
# trade frequency and avoid fee drag, with volume filter ensuring institutional participation.

name = "12h_KAMA_Direction_RSI20_80_VolumeFilter_V1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) - 2 periods for fast adaptation
    close_series = pd.Series(close)
    change = abs(close_series.diff(1))
    volatility = change.rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, 1e-10)  # Efficiency Ratio
    sc = (er * (0.66 - 0.06) + 0.06) ** 2  # Smoothing Constant
    kama = [close[0]]  # Initialize with first price
    for i in range(1, len(close)):
        kama.append(kama[-1] + sc[i] * (close[i] - kama[-1]))
    kama = np.array(kama)
    
    # Calculate RSI (14-period)
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA, RSI oversold (<20), volume confirmation
            if close[i] > kama[i] and rsi[i] < 20 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI overbought (>80), volume confirmation
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