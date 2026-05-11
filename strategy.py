#!/usr/bin/env python3
"""
1d_KAMA_RSI_ChopRegime
Hypothesis: Uses daily KAMA trend direction for bias, combined with RSI mean-reversion entries
and Choppiness Index regime filter to avoid trending markets. KAMA adapts to market noise,
providing reliable trend signals. RSI(14) < 30 for long, > 70 for short in ranging/choppy
markets (CHOP > 50). Works in bull markets by following KAMA trend and in bear markets by
fading extremes during range-bound periods. Designed for low trade frequency (~20-40/year)
by requiring trend alignment and regime filter.
"""

name = "1d_KAMA_RSI_ChopRegime"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily KAMA Trend Filter (ER=10, FA=2, SA=30) ---
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # will fix below
    
    # Correct efficiency ratio calculation
    er = np.zeros(n)
    price_change = np.abs(np.diff(close, n=10, prepend=close[:10]))
    volatility_sum = np.zeros(n)
    for i in range(10, n):
        volatility_sum[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
    er[10:] = price_change[10:] / np.where(volatility_sum[10:] == 0, 1, volatility_sum[10:])
    er = np.where(er > 1, 1, er)  # cap at 1
    
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # FA=2, SA=30
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # --- Daily RSI(14) ---
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # --- Daily Choppiness Index (CHOP) ---
    atr_period = 14
    tr1 = np.zeros(n)
    tr1[0] = high[0] - low[0]
    for i in range(1, n):
        tr1[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    for i in range(atr_period, n):
        atr[i] = np.mean(tr1[i-atr_period+1:i+1])
    
    # Max/min close over 14 periods
    max_close = np.zeros(n)
    min_close = np.zeros(n)
    for i in range(14, n):
        max_close[i] = np.max(close[i-13:i+1])
        min_close[i] = np.min(close[i-13:i+1])
    
    chop = np.full(n, 50.0)  # default to neutral
    for i in range(14, n):
        if max_close[i] - min_close[i] > 0:
            chop[i] = 100 * np.log10(np.sum(tr1[i-13:i+1]) / (max_close[i] - min_close[i])) / np.log10(14)
        else:
            chop[i] = 50.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    start_idx = 30  # need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Regime filter: only trade in choppy/ranging markets (CHOP > 50)
        choppy_market = chop[i] > 50
        
        if position == 0:
            # Long: RSI oversold in choppy market, price above KAMA (weak trend bias)
            if (rsi[i] < 30 and 
                choppy_market and 
                close[i] > kama[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought in choppy market, price below KAMA
            elif (rsi[i] > 70 and 
                  choppy_market and 
                  close[i] < kama[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: RSI returns to neutral range or regime shifts to trending
            if position == 1:
                if (rsi[i] > 50 or chop[i] <= 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if (rsi[i] < 50 or chop[i] <= 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals