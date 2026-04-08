#!/usr/bin/env python3
"""
12h KAMA + RSI + Chop Filter
Hypothesis: KAMA adapts to market noise, RSI identifies overbought/oversold, Chop filter avoids whipsaws.
Works in bull/bear by using adaptive trend + momentum + regime filter. Targets 15-35 trades/year on 12h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_kama_rsi_chop_filter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA (adaptive trend) - 12h
    change = np.abs(close - np.roll(close, 10))
    change[0:10] = 0
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute properly below
    # Proper ER calculation
    er = np.zeros_like(close)
    for i in range(10, n):
        if i >= 10:
            price_change = np.abs(close[i] - close[i-10])
            volatility_sum = np.sum(np.abs(np.diff(close[i-9:i+1])))
            if volatility_sum > 0:
                er[i] = price_change / volatility_sum
            else:
                er[i] = 0
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Chop Chop (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    
    # Volume filter (>1.3x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below KAMA OR RSI overbought OR chop high (trending end)
            if (close[i] <= kama[i] or 
                rsi[i] >= 70 or 
                chop[i] > 61.8):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above KAMA OR RSI oversold OR chop high
            if (close[i] >= kama[i] or 
                rsi[i] <= 30 or 
                chop[i] > 61.8):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price above KAMA, RSI oversold, chop low (range)
            if (close[i] > kama[i] and 
                rsi[i] < 30 and 
                chop[i] < 38.2 and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price below KAMA, RSI overbought, chop low (range)
            elif (close[i] < kama[i] and 
                  rsi[i] > 70 and 
                  chop[i] < 38.2 and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals