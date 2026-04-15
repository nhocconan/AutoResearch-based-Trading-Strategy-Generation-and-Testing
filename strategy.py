#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI + Choppiness Filter
# Uses Kaufman's Adaptive Moving Average (KAMA) to identify trend direction.
# Enters long when price > KAMA and RSI < 50 (pullback in uptrend).
# Enters short when price < KAMA and RSI > 50 (pullback in downtrend).
# Uses Choppiness Index (CHOP) to filter trades: only trade when CHOP < 50 (trending market).
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
# Target: 20-60 total trades over 4 years (5-15/year).
# Timeframe: 1d, HTF: 1w (for trend confirmation)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === KAMA (10-period) ===
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Handle first 10 values
    er = np.full_like(change, np.nan, dtype=float)
    er[10:] = change[10:] / volatility[10:]
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[9]  # Seed
    for i in range(10, n):
        if not np.isnan(sc[i-10]):
            kama[i] = kama[i-1] + sc[i-10] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # === RSI (14-period) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for first element
    rsi = np.concatenate([[np.nan], rsi])
    
    # === Choppiness Index (14-period) ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Choppiness
    chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(14)
    # Handle division by zero or invalid
    chop = np.where((hh - ll) > 0, chop, 50.0)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            continue
        
        # Long: price > KAMA (uptrend) + RSI < 50 (pullback) + CHOP < 50 (trending)
        if (close[i] > kama[i] and
            rsi[i] < 50 and
            chop[i] < 50 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short: price < KAMA (downtrend) + RSI > 50 (pullback) + CHOP < 50 (trending)
        elif (close[i] < kama[i] and
              rsi[i] > 50 and
              chop[i] < 50 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite signal or choppy market (CHOP >= 60)
        elif position == 1 and (close[i] < kama[i] or chop[i] >= 60):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > kama[i] or chop[i] >= 60):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_KAMA_RSI_Chop"
timeframe = "1d"
leverage = 1.0