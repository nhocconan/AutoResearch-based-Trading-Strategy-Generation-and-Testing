#!/usr/bin/env python3
# 4h_KAMA_Direction_Plus_RSI_With_Chop_Filter
# Hypothesis: Use KAMA to determine trend direction on 4h, combined with RSI for momentum confirmation
# and Choppiness Index to filter ranging markets. Enter long when KAMA trending up, RSI > 50, and market
# is trending (CHOP < 38.2); short when KAMA trending down, RSI < 50, and trending market.
# Designed for 20-40 trades per year by requiring trend alignment across multiple filters.
# Works in bull markets (follows KAMA trend) and bear markets (avoids false signals in chop via CHOP filter).

name = "4h_KAMA_Direction_Plus_RSI_With_Chop_Filter"
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
    
    # KAMA calculation (ER = 10)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # smooth constant
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14)
    atr = np.zeros_like(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((hh14 - ll14) != 0, 100 * np.log10(sum_atr14 / (hh14 - ll14)) / np.log10(14), 50)
    
    # Signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA trending up, RSI > 50, trending market (CHOP < 38.2)
            if (kama[i] > kama[i-1] and 
                rsi[i] > 50 and 
                chop[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # Short: KAMA trending down, RSI < 50, trending market (CHOP < 38.2)
            elif (kama[i] < kama[i-1] and 
                  rsi[i] < 50 and 
                  chop[i] < 38.2):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA turns down or RSI < 45
            if (kama[i] < kama[i-1] or 
                rsi[i] < 45):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA turns up or RSI > 55
            if (kama[i] > kama[i-1] or 
                rsi[i] > 55):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals