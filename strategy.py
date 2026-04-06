#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA direction + RSI + chop filter
# Long when KAMA indicates up trend AND RSI < 50 (pullback in uptrend) AND chop > 61.8 (range)
# Short when KAMA indicates down trend AND RSI > 50 (pullback in downtrend) AND chop > 61.8 (range)
# Exit when RSI crosses 50 in opposite direction
# Uses 12h timeframe to reduce trade frequency, KAMA for adaptive trend, RSI for pullback entries, chop for regime filter
# Target: 50-150 total trades over 4 years (12-37/year) for optimal 12h performance

name = "12h_kama_rsi_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA (adaptive trend)
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = abs(close - np.roll(close, 10))
    change[0:10] = 0  # first 10 bars have no 10-period change
    volatility = abs(np.diff(close, prepend=close[0]))
    er = change / pd.Series(volatility).rolling(window=10, min_periods=1).sum()
    er = er.fillna(0).values
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    # KAMA direction: 1 if kama rising, -1 if falling
    kama_dir = np.where(kama > np.roll(kama, 1), 1, -1)
    kama_dir[0] = 0
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Choppiness Index (14-period)
    atr = np.zeros_like(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10((atr * 14) / (highest_high - lowest_low)) / np.log10(14)
    chop = chop.fillna(50).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if required data not available
        if np.isnan(kama_dir[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: RSI crosses 50 in opposite direction
        if position == 1:  # long position
            if rsi[i] < 50 and rsi[i-1] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if rsi[i] > 50 and rsi[i-1] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with KAMA direction, RSI pullback, and chop filter
            # Long: KAMA up AND RSI < 50 (pullback) AND chop > 61.8 (range)
            if (kama_dir[i] == 1 and rsi[i] < 50 and chop[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Short: KAMA down AND RSI > 50 (pullback) AND chop > 61.8 (range)
            elif (kama_dir[i] == -1 and rsi[i] > 50 and chop[i] > 61.8):
                signals[i] = -0.25
                position = -1
    
    return signals