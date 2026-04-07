#!/usr/bin/env python3
"""
12h_kama_trend_1d_volume_v1
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing a reliable trend signal. On 12h timeframe, we use KAMA(10) for trend direction, confirmed by 1d EMA200 for higher timeframe alignment. Volume confirmation filters weak signals. This adapts to both trending and ranging markets, reducing false signals. Target: 12-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_kama_trend_1d_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA200 for trend filter
    ema_200 = df_1d['close'].ewm(span=200, adjust=False).mean()
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200.values)
    
    # KAMA(10) on 12h: Kaufman Adaptive Moving Average
    # Efficiency Ratio (ER) = |change| / volatility
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period sum of absolute changes
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start with close at index 9
    for i in range(10, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation (20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below KAMA or below 1d EMA200
            if close[i] < kama[i] or close[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above KAMA or above 1d EMA200
            if close[i] > kama[i] or close[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price above KAMA and above 1d EMA200, with volume
            if (close[i] > kama[i] and close[i] > ema_200_aligned[i] and vol_confirm):
                position = 1
                signals[i] = 0.25
            # Short entry: price below KAMA and below 1d EMA200, with volume
            elif (close[i] < kama[i] and close[i] < ema_200_aligned[i] and vol_confirm):
                position = -1
                signals[i] = -0.25
    
    return signals