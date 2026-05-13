#!/usr/bin/env python3
"""
1d_WMA_Price_Across_with_Volume_Confirmation
Hypothesis: Price crossing above/below a 21-period Weighted Moving Average (WMA)
with volume confirmation captures sustained trends in both bull and bear markets.
The WMA gives more weight to recent prices, making it responsive yet smooth.
Volume confirmation filters out weak breakouts. Designed for low trade frequency
(10-20/year) on daily timeframe to avoid fee drag and work across market regimes.
"""

name = "1d_WMA_Price_Across_with_Volume_Confirmation"
timeframe = "1d"
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
    
    # Calculate 21-period Weighted Moving Average (WMA)
    weights = np.arange(1, 22)
    wma = np.zeros(n)
    for i in range(21, n):
        wma[i] = np.dot(close[i-20:i+1], weights) / weights.sum()
    # For first 21 bars, use cumulative average of available data
    for i in range(21):
        wma[i] = np.mean(close[:i+1])
    
    # Volume confirmation: > 1.5x 20-period average volume
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    for i in range(20):
        vol_ma[i] = np.mean(volume[:i+1])
    volume_confirm = volume > (1.5 * vol_ma)
    
    # Get 1-week trend filter (EMA 50)
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(21, n):
        if position == 0:
            # LONG: Price crosses above WMA with volume confirmation
            if close[i] > wma[i] and close[i-1] <= wma[i-1] and volume_confirm[i]:
                # Additional filter: only take long if price above weekly EMA50 (uptrend filter)
                if close[i] > ema_50_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Price crosses below WMA with volume confirmation
            elif close[i] < wma[i] and close[i-1] >= wma[i-1] and volume_confirm[i]:
                # Additional filter: only take short if price below weekly EMA50 (downtrend filter)
                if close[i] < ema_50_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below WMA
            if close[i] < wma[i] and close[i-1] >= wma[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above WMA
            if close[i] > wma[i] and close[i-1] <= wma[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals