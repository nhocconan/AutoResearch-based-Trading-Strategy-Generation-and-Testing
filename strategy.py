#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d daily KAMA (14-period) + RSI(14) + Choppiness regime filter.
# KAMA adapts to market noise, capturing trend while avoiding whipsaws.
# RSI provides overbought/oversold signals in ranging markets.
# Choppiness index filters: >61.8 = range (mean revert), <38.2 = trend (follow trend).
# Designed for 1d timeframe to work in both bull and bear markets with low trade frequency.
# Entry: KAMA trending up + RSI < 40 in ranging market OR KAMA trending down + RSI > 60 in ranging market.
# Exit: Opposite KAMA direction or RSI crosses 50.
# Uses strict conditions to limit trades (~10-25/year) and avoid overtrading.

name = "1d_KAMA_RSI_Chop"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Kaufman Adaptive Moving Average (KAMA)
    def kama(close, period=14, fast=2, slow=30):
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.zeros_like(close)
        er[period:] = change[period-1:] / volatility[period-1:]
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[:period] = close[:period]
        for i in range(period, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_val = kama(close, 14, 2, 30)
    kama_dir = np.zeros_like(kama_val)
    kama_dir[1:] = np.where(kama_val[1:] > kama_val[:-1], 1, -1)
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    for i in range(15, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14-period)
    atr = np.zeros_like(close)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.insert(tr, 0, 0)
    atr_sum = np.zeros_like(close)
    for i in range(14, len(close)):
        atr_sum[i] = np.sum(tr[i-13:i+1])
    highest_high = np.zeros_like(close)
    lowest_low = np.zeros_like(close)
    for i in range(14, len(close)):
        highest_high[i] = np.max(high[i-13:i+1])
        lowest_low[i] = np.min(low[i-13:i+1])
    chop = np.zeros_like(close)
    for i in range(14, len(close)):
        if atr_sum[i] > 0 and highest_high[i] != lowest_low[i]:
            chop[i] = 100 * np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(14)
        else:
            chop[i] = 50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_val[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long in ranging market: KAMA up + RSI oversold
            if (kama_dir[i] == 1 and 
                chop[i] > 61.8 and 
                rsi[i] < 40):
                signals[i] = 0.25
                position = 1
            # Short in ranging market: KAMA down + RSI overbought
            elif (kama_dir[i] == -1 and 
                  chop[i] > 61.8 and 
                  rsi[i] > 60):
                signals[i] = -0.25
                position = -1
            # Follow trend in trending market: KAMA direction
            elif chop[i] < 38.2:
                if kama_dir[i] == 1:
                    signals[i] = 0.25
                    position = 1
                elif kama_dir[i] == -1:
                    signals[i] = -0.25
                    position = -1
                
        elif position == 1:
            # Long: exit if KAMA turns down OR RSI overbought
            if (kama_dir[i] == -1) or (rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if KAMA turns up OR RSI oversold
            if (kama_dir[i] == 1) or (rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals