#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Regime Filter
Long when KAMA is rising, RSI < 60, and chop > 61.8 (range) or chop < 38.2 (trend).
Short when KAMA is falling, RSI > 40, and chop > 61.8 or chop < 38.2.
Exit when KAMA direction reverses.
Designed for low turnover: ~10-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_period=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average"""
    close_series = pd.Series(close)
    change = abs(close_series.diff(er_period))
    volatility = close_series.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    return kama

def calculate_chop(high, low, close, period=14):
    """Choppiness Index"""
    atr = pd.DataFrame({'high': high, 'low': low, 'close': close})
    atr['tr0'] = atr['high'] - atr['low']
    atr['tr1'] = abs(atr['high'] - atr['close'].shift())
    atr['tr2'] = abs(atr['low'] - atr['close'].shift())
    atr['tr'] = atr[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr_sum = atr['tr'].rolling(window=period, min_periods=period).sum()
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    return chop.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate indicators
    kama = calculate_kama(close)
    rsi = pd.Series(close).rolling(window=14, min_periods=14).apply(
        lambda x: 100 - (100 / (1 + (x.diff().clip(min=0).mean() / (-x.diff().clip(max=0).mean()).replace(0, 1))))
    ).fillna(50).values
    chop = calculate_chop(high, low, close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(30, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            continue
        
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # Chop regime: >61.8 = range, <38.2 = trend
        chop_range = chop[i] > 61.8
        chop_trend = chop[i] < 38.2
        
        if position == 0:
            # Long: KAMA rising, RSI not overbought, in any regime
            if kama_rising and rsi[i] < 60 and (chop_range or chop_trend):
                position = 1
                signals[i] = position_size
            # Short: KAMA falling, RSI not oversold, in any regime
            elif kama_falling and rsi[i] > 40 and (chop_range or chop_trend):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: KAMA turns falling
            if kama_falling:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: KAMA turns rising
            if kama_rising:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_KAMA_RSI_Chop_Regime"
timeframe = "1d"
leverage = 1.0