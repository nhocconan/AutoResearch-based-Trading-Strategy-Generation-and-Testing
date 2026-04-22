#!/usr/bin/env python3
"""
Hypothesis: Daily KAMA trend with weekly RSI filter and volume confirmation.
Long when KAMA is rising, weekly RSI < 60, and volume > 1.5x average.
Short when KAMA is falling, weekly RSI > 40, and volume > 1.5x average.
Exit when KAMA reverses direction.
KAMA adapts to market noise, reducing whipsaws in ranging markets.
Weekly RSI prevents overextended entries. Volume confirms conviction.
Designed for low trade frequency (<15/year) with strong trend capture.
Works in bull markets via trend following and in bear markets via short signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (close, 10, 2, 30)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Load weekly data for RSI filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after enough data for indicators
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(kama[i-1]) or 
            np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA rising, weekly RSI < 60, volume > 1.5x average
            if (kama[i] > kama[i-1] and 
                rsi_1w_aligned[i] < 60 and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, weekly RSI > 40, volume > 1.5x average
            elif (kama[i] < kama[i-1] and 
                  rsi_1w_aligned[i] > 40 and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit when KAMA reverses direction
            if position == 1 and kama[i] < kama[i-1]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and kama[i] > kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "Daily_KAMA_WeeklyRSI_Volume_Filter"
timeframe = "1d"
leverage = 1.0