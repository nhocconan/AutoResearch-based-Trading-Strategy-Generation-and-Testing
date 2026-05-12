#!/usr/bin/env python3
"""
Hypothesis: 4h KAMA trend + RSI + Choppiness filter. Uses KAMA for adaptive trend, RSI for momentum exhaustion, and Choppiness to identify ranging markets where mean-reversion works. Designed to work in both bull and bear regimes by adapting to market conditions.
"""
name = "4h_KAMA_RSI_Chop_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === KAMA CALCULATION (10-period) ===
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.where(volatility > 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI (14-period) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === CHOPPINESS INDEX (14-period) ===
    atr = np.zeros_like(close)
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.abs(np.diff(close, prepend=close[0])))
    tr3 = np.abs(np.abs(np.diff(close, prepend=close[0])) - np.abs(high - low))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.where((highest_high - lowest_low) > 0,
                    100 * np.log10(np.sum(atr, axis=0) / (highest_high - lowest_low)) / np.log10(14),
                    50)
    
    # Handle NaN in chop calculation
    chop = np.where(np.isnan(chop), 50, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # KAMA needs ~30, RSI/CHOP need 14
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above KAMA (uptrend), RSI not overbought, choppy market (mean reversion)
            if (close[i] > kama[i] and 
                rsi[i] < 70 and 
                chop[i] > 50):  # Choppy/ranging market
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA (downtrend), RSI not oversold, choppy market
            elif (close[i] < kama[i] and 
                  rsi[i] > 30 and 
                  chop[i] > 50):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price below KAMA or RSI overbought
            if (close[i] < kama[i]) or (rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above KAMA or RSI oversold
            if (close[i] > kama[i]) or (rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals