#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_Weekly_RSI_Filter
Hypothesis: KAMA trend direction on 1d filtered by weekly RSI (70/30) and volume confirmation on 1d. 
KAMA adapts to market noise, reducing false signals in sideways markets. Weekly RSI avoids 
overbought/oversold extremes, working in both bull (buy dips in uptrend) and bear (sell rallies 
in downtrend). Volume confirmation ensures breakouts have conviction. Target: 15-25 trades/year.
"""

name = "1d_KAMA_Trend_With_Weekly_RSI_Filter"
timeframe = "1d"
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
    
    # Get weekly data ONCE before loop
    df_w = get_htf_data(prices, '1w')
    
    # Calculate KAMA on daily close
    def kama(close, er_length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_vals = kama(close)
    
    # Weekly RSI (14)
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_w = rsi(df_w['close'].values)
    rsi_w_aligned = align_htf_to_ltf(prices, df_w, rsi_w)
    
    # Volume confirmation: >1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above KAMA (uptrend) + weekly RSI < 70 (not overbought) + volume spike
            if (close[i] > kama_vals[i] and 
                rsi_w_aligned[i] < 70 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA (downtrend) + weekly RSI > 30 (not oversold) + volume spike
            elif (close[i] < kama_vals[i] and 
                  rsi_w_aligned[i] > 30 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA (trend change)
            if close[i] < kama_vals[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA (trend change)
            if close[i] > kama_vals[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals