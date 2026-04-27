#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA (Adaptive Moving Average) with RSI and chop filter.
# Uses KAMA trend direction (above/below) + RSI(14) for momentum + Choppiness Index for regime.
# Long when KAMA rising, RSI > 50, and CHOP < 40 (trending market).
# Short when KAMA falling, RSI < 50, and CHOP < 40.
# Exit when CHOP > 60 (choppy market) or RSI crosses 50 opposite.
# Aims for 20-40 trades/year by requiring trend + momentum + low chop.
# KAMA adapts to market noise, reducing false signals in chop.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    er_length = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=1))
    change = np.insert(change, 0, 0)  # align length
    abs_change = np.abs(np.diff(close, n=1))
    abs_change = np.insert(abs_change, 0, 0)
    
    # ER = |net change| / sum(abs changes) over er_length period
    net_change = np.zeros(n)
    total_change = np.zeros(n)
    for i in range(er_length, n):
        net_change[i] = np.abs(close[i] - close[i-er_length])
        total_change[i] = np.sum(abs_change[i-er_length+1:i+1])
    
    # Avoid division by zero
    er = np.zeros(n)
    er[total_change != 0] = net_change[total_change != 0] / total_change[total_change != 0]
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros(n)
    rs[avg_loss != 0] = avg_gain[avg_loss != 0] / avg_loss[avg_loss != 0]
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100  # when no loss, RSI=100
    
    # Choppiness Index (14-period)
    atr = np.zeros(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0]-low[0], np.abs(high[0]-close[0]), np.abs(low[0]-close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    hh = np.zeros(n)
    ll = np.zeros(n)
    for i in range(14, n):
        hh[i] = np.max(high[i-13:i+1])
        ll[i] = np.min(low[i-13:i+1])
    
    # Chop = 100 * log10(sum(ATR14) / (HH - LL)) / log10(14)
    chop = np.zeros(n)
    for i in range(14, n):
        if hh[i] > ll[i]:
            sum_atr = np.sum(atr[i-13:i+1])
            chop[i] = 100 * np.log10(sum_atr / (hh[i] - ll[i])) / np.log10(14)
        else:
            chop[i] = 50  # default when range is zero
    
    # Signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Long: KAMA up, RSI > 50, low chop (trending)
        if kama[i] > kama[i-1] and rsi[i] > 50 and chop[i] < 40:
            signals[i] = 0.25
            position = 1
        # Short: KAMA down, RSI < 50, low chop (trending)
        elif kama[i] < kama[i-1] and rsi[i] < 50 and chop[i] < 40:
            signals[i] = -0.25
            position = -1
        # Exit: choppy market or RSI crosses 50 opposite
        elif position == 1 and (chop[i] > 60 or rsi[i] < 50):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (chop[i] > 60 or rsi[i] > 50):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_KAMA_RSI_Chop_Filter"
timeframe = "4h"
leverage = 1.0