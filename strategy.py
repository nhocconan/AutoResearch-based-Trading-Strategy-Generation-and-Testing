#!/usr/bin/env python3
"""
12h_KAMA_Trend_with_RSI_Filter_and_ATR_Stop
Hypothesis: KAMA adapts to market efficiency, providing robust trend signals in both bull and bear regimes.
RSI filter avoids overextended entries, while ATR-based stop manages risk. Designed for low-frequency,
high-conviction trades on 12h timeframe to minimize fee drag (target: 12-37 trades/year).
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
    
    # KAMA (Kaufman Adaptive Moving Average) - adaptive trend filter
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    er = np.zeros(n)
    for i in range(10, n):  # ER period = 10
        if np.sum(volatility[i-9:i+1]) > 0:
            er[i] = np.abs(close[i] - close[i-10]) / np.sum(volatility[i-9:i+1])
        else:
            er[i] = 0
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # fast=2, slow=30
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) for overbought/oversold filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # ATR(20) for volatility-based stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    entry_price = 0
    
    start_idx = max(20, 14, 10)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price above KAMA and RSI not overbought
            if price > kama[i] and rsi[i] < 70:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price below KAMA and RSI not oversold
            elif price < kama[i] and rsi[i] > 30:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses below KAMA or ATR-based stop
            if price < kama[i] or price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses above KAMA or ATR-based stop
            if price > kama[i] or price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_KAMA_Trend_with_RSI_Filter_and_ATR_Stop"
timeframe = "12h"
leverage = 1.0