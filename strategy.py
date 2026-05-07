#!/usr/bin/env python3
name = "4h_KAMA_Direction_RSI_ChopFilter"
timeframe = "4h"
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
    
    # KAMA parameters
    fast = 2
    slow = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.zeros(n)
    for i in range(n):
        if i == 0:
            er[i] = 0
        else:
            change_sum = np.sum(change[max(0, i-9):i+1])
            volatility_sum = np.sum(np.abs(np.diff(close[max(0, i-9):i+1])))
            er[i] = change_sum / (volatility_sum + 1e-10)
    
    # Smoothing constants
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction: 1 for up, -1 for down
    kama_dir = np.where(kama > np.roll(kama, 1), 1, -1)
    kama_dir[0] = 0
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14)
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    sum_atr14 = np.zeros(n)
    for i in range(14, n):
        sum_atr14[i] = np.sum(atr[i-13:i+1])
    
    hh = np.zeros(n)
    ll = np.zeros(n)
    for i in range(n):
        if i < 14:
            hh[i] = high[i]
            ll[i] = low[i]
        else:
            hh[i] = np.max(high[i-13:i+1])
            ll[i] = np.min(low[i-13:i+1])
    
    chop = np.zeros(n)
    for i in range(14, n):
        if hh[i] != ll[i]:
            chop[i] = 100 * np.log10(sum_atr14[i] / (hh[i] - ll[i])) / np.log10(14)
        else:
            chop[i] = 50
    
    # Chop filter: chop > 61.8 = range (mean revert), chop < 38.2 = trending (trend follow)
    chop_range = chop > 61.8
    chop_trend = chop < 38.2
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(20, min_periods=1).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if KAMA not ready
        if np.isnan(kama[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA up + RSI > 50 + chop trend + volume confirmation
            if (kama_dir[i] == 1 and 
                rsi[i] > 50 and 
                chop_trend[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA down + RSI < 50 + chop trend + volume confirmation
            elif (kama_dir[i] == -1 and 
                  rsi[i] < 50 and 
                  chop_trend[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: KAMA down OR RSI < 40
            if kama_dir[i] == -1 or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: KAMA up OR RSI > 60
            if kama_dir[i] == 1 or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: On 4h timeframe, KAMA direction captures adaptive trend, RSI filters momentum strength, and Chop regime ensures we only trade in trending markets. Volume confirmation adds institutional validation. Works in bull markets (KAMA up + RSI > 50) and bear markets (KAMA down + RSI < 50). Discrete position sizing (0.25) limits drawdown. Target: 40-100 trades/year to minimize fee drag. Combines trend (KAMA), momentum (RSI), and regime (Chop) for robust performance across market cycles.