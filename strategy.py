#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_Volume_And_Chop_Filter
Hypothesis: KAMA adapts to market noise, providing reliable trend signals in both trending and ranging markets.
Combined with volume confirmation (>1.5x 20-period average) and Choppiness Index (<40 for trending, >60 for ranging),
this strategy avoids whipsaws. Long when KAMA rising + volume + chop<40; short when KAMA falling + volume + chop>60.
Position size 0.25 targets ~20-30 trades/year to minimize fee drag.
"""

name = "4h_KAMA_Trend_With_Volume_And_Chop_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    close = pd.Series(close)
    change = abs(close - close.shift(er_period))
    volatility = abs(close.diff()).rolling(window=er_period).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = [np.nan] * len(close)
    kama[0] = close.iloc[0]
    for i in range(1, len(close)):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close.iloc[i] - kama[i-1])
    return np.array(kama)

def calculate_chop(high, low, close, period=14):
    """Calculate Choppiness Index"""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    atr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    tr_sum = atr.rolling(window=period).sum()
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    chop = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    return chop.fillna(50).values  # fill NaN with neutral 50

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA trend (ER=10, fast=2, slow=30)
    kama = calculate_kama(close, er_period=10, fast=2, slow=30)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Choppiness Index for regime filter
    chop = calculate_chop(high, low, close, period=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup for KAMA and CHOP
        if position == 0:
            # LONG: KAMA rising, volume confirmation, trending market (CHOP < 40)
            if (kama[i] > kama[i-1] and 
                volume_filter[i] and 
                chop[i] < 40):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling, volume confirmation, ranging market (CHOP > 60)
            elif (kama[i] < kama[i-1] and 
                  volume_filter[i] and 
                  chop[i] > 60):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns down OR chop becomes too high (ranging)
            if (kama[i] < kama[i-1]) or (chop[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up OR chop becomes too low (trending)
            if (kama[i] > kama[i-1]) or (chop[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals