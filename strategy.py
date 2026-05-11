#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI_Range_Entry
Hypothesis: Use 4h KAMA to determine trend direction and RSI for mean-reversion entries in ranging markets.
In bull/bear markets, KAMA filters trend direction; in ranges, RSI extremes with KAMA flat provide entries.
Combines trend-following and mean-reversion with volume confirmation to reduce false signals.
Targets 20-40 trades/year on 4h timeframe with low frequency to minimize fee drag.
"""

name = "4h_KAMA_Trend_RSI_Range_Entry"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === KAMA Trend Filter (4h) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    er = np.concatenate([np.full(10, np.nan), er])
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI (14) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Volume Filter (1.5x 20-period SMA) ===
    vol_sma20 = np.full_like(volume, np.nan)
    for i in range(20, n):
        vol_sma20[i] = np.mean(volume[i-20:i])
    volume_ok = volume > vol_sma20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Enter long: RSI oversold in ranging market (KAMA flat)
            if (rsi[i] < 30 and 
                np.abs(close[i] - kama[i]) / kama[i] < 0.02 and  # price near KAMA
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: RSI overbought in ranging market
            elif (rsi[i] > 70 and 
                  np.abs(close[i] - kama[i]) / kama[i] < 0.02 and
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
            # Trend following: KAMA slope confirms trend
            elif i >= 2:
                kama_slope = (kama[i] - kama[i-2]) / 2
                if (kama_slope > 0 and  # uptrend
                    close[i] > kama[i] and
                    volume_ok[i]):
                    signals[i] = 0.25
                    position = 1
                elif (kama_slope < 0 and  # downtrend
                      close[i] < kama[i] and
                      volume_ok[i]):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: RSI overbought or price below KAMA
            if rsi[i] > 70 or close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI oversold or price above KAMA
            if rsi[i] < 30 or close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals