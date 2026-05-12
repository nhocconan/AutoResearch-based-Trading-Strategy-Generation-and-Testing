#!/usr/bin/env python3
# 1d_KAMA_Direction_RSI_Chop_Filter
# Hypothesis: 1d KAMA direction with RSI(2) and Choppiness Index filter for mean reversion in chop and trend following in trends.
# KAMA adapts to market noise, RSI(2) captures short-term extremes, Choppiness Index filters regime.
# Works in bull markets via trend following (KAMA direction) and in bear/chop via mean reversion (RSI extremes).
# Volume confirmation ensures signal validity.

name = "1d_KAMA_Direction_RSI_Chop_Filter"
timeframe = "1d"
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
    
    # === 1d KAMA (ER=10, FAST=2, SLOW=30) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute properly below
    # Recompute volatility as rolling sum of absolute changes
    volatility = pd.Series(change).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI(2) ===
    delta = np.diff(close, prepend=close[0])
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(up).rolling(window=2, min_periods=2).mean()
    roll_down = pd.Series(down).rolling(window=2, min_periods=2).mean()
    rs = np.where(roll_down > 0, roll_up / roll_down, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppiness Index (14) ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    # Sum of True Range over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max()
    ll = pd.Series(low).rolling(window=14, min_periods=14).min()
    # Chop calculation
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    # Avoid division by zero or invalid
    chop = np.where((hh - ll) > 0, chop, 50)
    
    # === Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # KAMA direction: price above/below KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI extremes
        rsi_oversold = rsi[i] < 15
        rsi_overbought = rsi[i] > 85
        
        # Chop regime: chop > 61.8 = ranging (mean revert), chop < 38.2 = trending
        chop_ranging = chop[i] > 61.8
        chop_trending = chop[i] < 38.2
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: In ranging market, RSI oversold + volume
            # OR in trending market, price above KAMA + volume
            if (chop_ranging and rsi_oversold and vol_ok) or (chop_trending and price_above_kama and vol_ok):
                signals[i] = 0.25
                position = 1
            # SHORT: In ranging market, RSI overbought + volume
            # OR in trending market, price below KAMA + volume
            elif (chop_ranging and rsi_overbought and vol_ok) or (chop_trending and price_below_kama and vol_ok):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: RSI overbought or price below KAMA in trending market
            if rsi[i] > 70 or (chop_trending and not price_above_kama):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI oversold or price above KAMA in trending market
            if rsi[i] < 30 or (chop_trending and not price_below_kama):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals