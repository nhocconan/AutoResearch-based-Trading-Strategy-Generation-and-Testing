#!/usr/bin/env python3
# 4h_KAMA_Trend_RSI_Extremes_ChopFilter_v3
# Hypothesis: KAMA trend filter + RSI extremes + Choppiness regime filter on 4h.
# In trending markets (CHOP < 38.2), follow KAMA direction. In ranging markets (CHOP > 61.8), fade RSI extremes.
# Avoids whipsaws by requiring regime alignment. Works in bull/bear by adapting to market structure.
# Uses 20-period RSI with 30/70 extremes and 14-period Choppiness Index.

name = "4h_KAMA_Trend_RSI_Extremes_ChopFilter_v3"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from math import fabs

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate ER (Efficiency Ratio) for KAMA
    change = np.zeros(n)
    for i in range(1, n):
        change[i] = fabs(close[i] - close[i-1])
    
    ir = np.zeros(n)
    for i in range(10, n):  # 10-period ER
        ir[i] = fabs(close[i] - close[i-10])
    
    er = np.zeros(n)
    for i in range(10, n):
        sum_change = np.sum(change[i-9:i+1])  # sum of last 10 changes
        if sum_change > 0:
            er[i] = ir[i] / sum_change
        else:
            er[i] = 0
    
    # Smooth constants
    sc = (er * 0.6 + 0.064) ** 2  # fast SC = 2/(2+1), slow SC = 2/(30+1)
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI (14-period)
    delta = np.zeros(n)
    for i in range(1, n):
        delta[i] = close[i] - close[i-1]
    
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
    rsi = np.zeros(n)
    for i in range(14, n):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100
    
    # Calculate Choppiness Index (14-period)
    atr = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], fabs(high[i] - close[i-1]), fabs(low[i] - close[i-1]))
    
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    max_high = np.zeros(n)
    min_low = np.zeros(n)
    for i in range(14, n):
        max_high[i] = np.max(high[i-13:i+1])
        min_low[i] = np.min(low[i-13:i+1])
    
    chop = np.zeros(n)
    for i in range(14, n):
        if atr[i] > 0 and (max_high[i] - min_low[i]) > 0:
            chop[i] = 100 * np.log10(sum(tr[i-13:i+1]) / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine market regime
            if chop[i] < 38.2:  # Trending market
                # Enter long: price above KAMA AND RSI > 50 (bullish bias)
                if close[i] > kama[i] and rsi[i] > 50:
                    signals[i] = 0.25
                    position = 1
                # Enter short: price below KAMA AND RSI < 50 (bearish bias)
                elif close[i] < kama[i] and rsi[i] < 50:
                    signals[i] = -0.25
                    position = -1
            elif chop[i] > 61.8:  # Ranging market
                # Enter long: RSI oversold (<30) AND price near support
                if rsi[i] < 30 and close[i] <= kama[i] * 1.02:  # near KAMA as support
                    signals[i] = 0.25
                    position = 1
                # Enter short: RSI overbought (>70) AND price near resistance
                elif rsi[i] > 70 and close[i] >= kama[i] * 0.98:  # near KAMA as resistance
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: trend change or RSI overbought in range
            if chop[i] < 38.2:  # trending
                if close[i] < kama[i]:  # trend reversal
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # ranging
                if rsi[i] > 70:  # overbought exit
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend change or RSI oversold in range
            if chop[i] < 38.2:  # trending
                if close[i] > kama[i]:  # trend reversal
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # ranging
                if rsi[i] < 30:  # oversold exit
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals