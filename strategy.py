#!/usr/bin/env python3
"""
1d_KAMA_With_RSI_And_Chop_Regime
Uses daily KAMA for trend direction, RSI(14) for momentum, and Choppiness Index for regime filtering.
Enters long when KAMA is rising, RSI(14) > 50, and Choppiness Index > 61.8 (range regime).
Enters short when KAMA is falling, RSI(14) < 50, and Choppiness Index > 61.8.
Exits when KAMA reverses direction or RSI crosses 50 in opposite direction.
Designed for low trade frequency (~50-80 total trades over 4 years) to minimize fee drift.
Works in both bull and bear markets by using regime filter to avoid trending whipsaws.
"""

name = "1d_KAMA_With_RSI_And_Chop_Regime"
timeframe = "1d"
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
    
    # KAMA: Kaufman Adaptive Moving Average
    # ER = Efficiency Ratio = |close - close[10]| / sum(|close - close[-1]| over 10 periods)
    # SC = [ER * (fastest - slowest) + slowest]^2
    # KAMA = prevKAMA + SC * (price - prevKAMA)
    change = np.abs(np.diff(close, n=10))  # |close - close[10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=1)  # sum of abs changes over 10 periods
    # Pad arrays to match length
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smooth ER
    er_smoothed = pd.Series(er).ewm(alpha=2/(2+1), adjust=False).fillna(0).values  # EMA of ER
    fastest = 2/(2+1)  # for EMA(2)
    slowest = 2/(30+1) # for EMA(30)
    sc = (er_smoothed * (fastest - slowest) + slowest) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad RSI to match length (first 14 values are NaN)
    rsi = np.concatenate([np.full(14, np.nan), rsi[14:]])
    
    # Choppiness Index (14)
    # ATR(14)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with close
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Max/Min over 14 periods
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(ATR/14) / (max_high - min_low)) / log10(14)
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    denominator = max_high - min_low
    chop = np.where(denominator != 0, 
                    100 * np.log10(sum_atr / denominator) / np.log10(14), 
                    50)  # default to neutral when no range
    
    # KAMA direction: 1 if rising, -1 if falling, 0 if flat
    kama_dir = np.where(kama > np.roll(kama, 1), 1, 
                        np.where(kama < np.roll(kama, 1), -1, 0))
    kama_dir[0] = 0  # first value
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # start after RSI and Chop warmup
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(kama_dir[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: KAMA rising, RSI > 50, Chop > 61.8 (range)
            if (kama_dir[i] == 1 and 
                rsi[i] > 50 and 
                chop[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling, RSI < 50, Chop > 61.8 (range)
            elif (kama_dir[i] == -1 and 
                  rsi[i] < 50 and 
                  chop[i] > 61.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA falling OR RSI < 50
            if (kama_dir[i] == -1) or (rsi[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA rising OR RSI > 50
            if (kama_dir[i] == 1) or (rsi[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals