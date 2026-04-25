#!/usr/bin/env python3
"""
12h_KAMA_Direction_1wTrend_RegimeFilter_v1
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) direction on 12h with 1w EMA trend filter and 12h choppiness regime filter.
Enter long when KAMA turns upward (price > KAMA) and 1w trend is bullish, but only in trending regimes (CHOP < 38.2).
Enter short when KAMA turns downward (price < KAMA) and 1w trend is bearish, but only in trending regimes (CHOP < 38.2).
Exit on opposite KAMA signal or regime shift to choppy (CHOP > 61.8).
Position size: 0.25 to limit drawdown.
Target: 12-25 trades/year (50-100 total over 4 years) to stay well under 200-trade 12h hard max.
Works in bull (KAMA up with uptrend) and bear (KAMA down with downtrend) markets by filtering with 1w trend and avoiding whipsaws via chop regime.
"""

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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for EMA30
        return np.zeros(n)
    
    # Calculate 1w EMA30 for HTF trend filter
    close_1w = df_1w['close'].values
    ema_30_1w = pd.Series(close_1w).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_30_1w)
    
    # Calculate KAMA on 12h
    # Efficiency ratio: ER = |close - close[10]| / sum(|close - close[1]|, 10)
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.zeros(n)
    for i in range(10, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))  # sum |diff| over 10 periods
    # Avoid division by zero
    er = np.zeros(n)
    er[10:] = change[10:] / np.where(volatility[10:] == 0, 1, volatility[10:])
    # Smoothing constants: fastest EMA(2), slowest EMA(30)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # = [ER*(0.5 - 0.0333) + 0.0333]^2
    # Calculate KAMA
    kama = np.zeros(n)
    kama[:10] = close[:10]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate Choppiness Index on 12h (CHOP)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    # Sum of TR over 14 periods
    atr14 = np.zeros(n)
    for i in range(14, n):
        atr14[i] = np.sum(tr[i-13:i+1])
    # Max and min close over 14 periods
    max_close = np.zeros(n)
    min_close = np.zeros(n)
    for i in range(14, n):
        max_close[i] = np.max(close[i-13:i+1])
        min_close[i] = np.min(close[i-13:i+1])
    # CHOP = 100 * log10(sum(TR14) / (max_close - min_close)) / log10(14)
    chop = np.zeros(n)
    for i in range(14, n):
        if max_close[i] > min_close[i]:
            chop[i] = 100 * np.log10(atr14[i] / (max_close[i] - min_close[i])) / np.log10(14)
        else:
            chop[i] = 50  # undefined, set to middle
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for KAMA (10) and CHOP (14)
    start_idx = 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_30_1w_aligned[i]) or 
            np.isnan(kama[i]) or np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend (bullish = price above EMA30)
        htf_1w_bullish = close[i] > ema_30_1w_aligned[i]
        htf_1w_bearish = close[i] < ema_30_1w_aligned[i]
        
        # Regime filter: trending when CHOP < 38.2, choppy when CHOP > 61.8
        is_trending = chop[i] < 38.2
        is_choppy = chop[i] > 61.8
        
        if position == 0:
            # Long setup: KAMA turning up (price > KAMA) + 1w uptrend + trending regime
            long_setup = (close[i] > kama[i]) and htf_1w_bullish and is_trending
            
            # Short setup: KAMA turning down (price < KAMA) + 1w downtrend + trending regime
            short_setup = (close[i] < kama[i]) and htf_1w_bearish and is_trending
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price < KAMA (reverse) OR regime turns choppy
            if (close[i] < kama[i]) or is_choppy:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price > KAMA (reverse) OR regime turns choppy
            if (close[i] > kama[i]) or is_choppy:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_KAMA_Direction_1wTrend_RegimeFilter_v1"
timeframe = "12h"
leverage = 1.0