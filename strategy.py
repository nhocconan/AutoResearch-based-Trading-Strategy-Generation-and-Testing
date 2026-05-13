#!/usr/bin/env python3
"""
12h_KAMA_RSI_Trend_With_Volume_Filter
Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both bull and bear markets. A long signal triggers when price crosses above KAMA with RSI > 50 and volume confirmation; short when price crosses below KAMA with RSI < 50 and volume confirmation. Uses 1d trend filter (price > EMA50) to align with higher timeframe trend. Position size 0.25 to limit frequency (~15-30/year) and reduce fee drag.
"""

name = "12h_KAMA_RSI_Trend_With_Volume_Filter"
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
    
    # KAMA parameters
    er_length = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0])).cumsum() - np.abs(np.diff(close, prepend=close[0])).rolling(window=er_length, min_periods=er_length).sum()
    er = change / (volatility + 1e-10)
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        if position == 0:
            # LONG: price crosses above KAMA, RSI > 50, volume confirmation, price above 1d EMA50 (uptrend)
            if (close[i] > kama[i] and close[i-1] <= kama[i-1] and 
                rsi_values[i] > 50 and 
                volume_filter[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price crosses below KAMA, RSI < 50, volume confirmation, price below 1d EMA50 (downtrend)
            elif (close[i] < kama[i] and close[i-1] >= kama[i-1] and 
                  rsi_values[i] < 50 and 
                  volume_filter[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below KAMA OR RSI < 40
            if (close[i] < kama[i] and close[i-1] >= kama[i-1]) or \
               rsi_values[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above KAMA OR RSI > 60
            if (close[i] > kama[i] and close[i-1] <= kama[i-1]) or \
               rsi_values[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals