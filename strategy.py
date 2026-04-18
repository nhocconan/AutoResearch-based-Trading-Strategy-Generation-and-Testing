#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_Chop
Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) for trend direction on 1d, combined with RSI for momentum and Choppiness Index for regime filtering. Only takes trades when KAMA slope aligns with RSI extremes and market is trending (CHOP < 38.2) or mean-reverting (CHOP > 61.8) with appropriate RSI conditions. Designed for low-frequency trading (7-25 trades/year) to minimize fee drift while capturing major trends and reversals in both bull and bear markets.
"""

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === INDICATOR CALCULATIONS ===
    # KAMA (10-period ER, 2/30 fast/slow)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # will compute properly below
    # Recalculate volatility as rolling sum of absolute changes
    volatility = np.zeros(n)
    for i in range(1, n):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
        if i >= 10:
            volatility[i] -= np.abs(close[i-10] - close[i-11] if i-11 >= 0 else 0)
    er = np.zeros(n)
    er[9:] = np.abs(close[9:] - close[:-9]) / np.where(volatility[9:] != 0, volatility[9:], 1)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(1, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14 if i >= 14 else np.mean(gain[max(0, i-13):i+1])
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14 if i >= 14 else np.mean(loss[max(0, i-13):i+1])
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14-period)
    atr = np.zeros(n)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14 if i >= 14 else np.mean(tr[max(0, i-13):i+1])
    atr_sum = np.zeros(n)
    for i in range(1, n):
        atr_sum[i] = atr_sum[i-1] + atr[i]
        if i >= 14:
            atr_sum[i] -= atr[i-14]
    highest_high = np.zeros(n)
    lowest_low = np.zeros(n)
    for i in range(n):
        highest_high[i] = np.max(high[max(0, i-13):i+1])
        lowest_low[i] = np.min(low[max(0, i-13):i+1])
    chop = np.zeros(n)
    for i in range(13, n):
        if atr_sum[i] > 0 and (highest_high[i] - lowest_low[i]) > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(14)
        else:
            chop[i] = 50.0
    
    # === SIGNAL LOGIC ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # KAMA slope (direction)
        kama_slope = kama[i] - kama[i-1]
        
        if position == 0:
            # Long conditions: KAMA rising AND (RSI oversold in chop OR RSI strong in trend)
            if kama_slope > 0:
                if (chop[i] > 61.8 and rsi[i] < 30) or (chop[i] <= 61.8 and rsi[i] > 50):
                    signals[i] = 0.25
                    position = 1
            # Short conditions: KAMA falling AND (RSI overbought in chop OR RSI weak in trend)
            elif kama_slope < 0:
                if (chop[i] > 61.8 and rsi[i] > 70) or (chop[i] <= 61.8 and rsi[i] < 50):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: KAMA turns down OR RSI overbought in trend
            if kama_slope < 0 or (chop[i] <= 61.8 and rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA turns up OR RSI oversold in trend
            if kama_slope > 0 or (chop[i] <= 61.8 and rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_RSI_Chop"
timeframe = "1d"
leverage = 1.0