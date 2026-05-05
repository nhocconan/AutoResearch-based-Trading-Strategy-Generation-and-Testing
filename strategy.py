#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + RSI(14) mean reversion + choppiness regime filter
# Long when: KAMA rising AND RSI < 30 AND Chop > 61.8 (range) → mean reversion long in range
# Short when: KAMA falling AND RSI > 70 AND Chop > 61.8 (range) → mean reversion short in range
# Exit: RSI crosses 50 (mean reversion complete) OR Chop < 38.2 (trend regime) → follow KAMA
# Uses KAMA for adaptive trend, RSI for overextension, Chop for regime detection
# Timeframe: 1d, HTF: 1w for trend confirmation (optional enhancement)
# Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag

name = "1d_KAMA_RSI_Chop_MeanReversion"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (adaptive moving average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[t] - close[t-1]| over 10 periods
    # Fix array alignment: volatility needs same length as change
    volatility = pd.Series(close).rolling(window=10, min_periods=10).apply(lambda x: np.sum(np.abs(np.diff(x))), raw=True).values
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants: fastest EMA=2, slowest EMA=30
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # (ER * (fast - slow) + slow)^2
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # start at first valid ER
    for i in range(10, n):
        if not np.isnan(sc[i-10]):  # sc index offset by 10
            kama[i] = kama[i-1] + sc[i-10] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI(14)
    if len(close) >= 15:
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
        avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        # Prepend NaN for first element
        rsi = np.concatenate([np.full(1, np.nan), rsi])
    else:
        rsi = np.full(n, np.nan)
    
    # Calculate Choppiness Index(14)
    if len(high) >= 14 and len(low) >= 14 and len(close) >= 14:
        atr = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))).rolling(window=14, min_periods=14).mean().values
        max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        chop = 100 * np.log10(atr * 14 / (max_high - min_low)) / np.log10(14)
        # Handle division by zero or invalid values
        chop = np.where((max_high - min_low) > 0, chop, 50.0)
    else:
        chop = np.full(n, 50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA rising (trend up) AND RSI oversold (<30) AND choppy market (>61.8)
            if (kama[i] > kama[i-1] and 
                rsi[i] < 30 and 
                chop[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling (trend down) AND RSI overbought (>70) AND choppy market (>61.8)
            elif (kama[i] < kama[i-1] and 
                  rsi[i] > 70 and 
                  chop[i] > 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI crosses above 50 (mean reversion complete) OR chop < 38.2 (trending regime)
            if (rsi[i] > 50 or chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI crosses below 50 (mean reversion complete) OR chop < 38.2 (trending regime)
            if (rsi[i] < 50 or chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals