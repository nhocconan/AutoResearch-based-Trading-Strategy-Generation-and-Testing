#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA with RSI and Chop Regime Filter
# Uses Kaufman Adaptive Moving Average (KAMA) to capture adaptive trend.
# Long when price > KAMA and RSI < 70 (avoid overbought), short when price < KAMA and RSI > 30 (avoid oversold).
# Chop regime filter: only trade when Choppiness Index > 61.8 (ranging market) for mean reversion.
# Works in bull markets (buy dips in range) and bear markets (sell rallies in range).
# Target: 30-100 total trades over 4 years (7-25/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (10-period ER, 2 and 30 for fast/slow SC)
    # Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # Will fix below
    # Recalculate volatility properly
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    # Actually compute rolling volatility
    volatility = np.zeros(n)
    for i in range(1, n):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    # Correct way: sum of absolute changes over ER period
    er_period = 10
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility_sum = np.zeros(n)
    for i in range(er_period, n):
        volatility_sum[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    er = np.where(volatility_sum > 0, change / volatility_sum, 0)
    # Smoothing Constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    # KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    # Sum of TR over period
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over period
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Choppiness
    chop = np.where((hh - ll) > 0, 
                    100 * np.log10(tr_sum / (hh - ll)) / np.log10(14), 
                    50)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(14, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            continue
        
        # Long: price > KAMA, RSI < 70 (not overbought), chop > 61.8 (ranging)
        if (close[i] > kama[i] and
            rsi[i] < 70 and
            chop[i] > 61.8 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short: price < KAMA, RSI > 30 (not oversold), chop > 61.8 (ranging)
        elif (close[i] < kama[i] and
              rsi[i] > 30 and
              chop[i] > 61.8 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or chop < 38.2 (trending market)
        elif position == 1 and (close[i] < kama[i] or chop[i] < 38.2):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > kama[i] or chop[i] < 38.2):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_KAMA_RSI_Chop_Regime"
timeframe = "1d"
leverage = 1.0