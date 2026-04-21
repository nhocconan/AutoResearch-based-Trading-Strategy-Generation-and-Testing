#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA (10-period ER) direction + RSI(14) + 12h chop filter (Choppiness Index > 61.8)
# Long when KAMA rising, RSI > 50, and market is choppy (mean-reversion prone)
# Short when KAMA falling, RSI < 50, and market is choppy
# Uses KAMA for adaptive trend, RSI for momentum, chop filter to avoid trending markets
# Designed for range-bound markets (2025-2026) while avoiding whipsaws in strong trends
# Target: 20-40 trades/year by requiring all three conditions to align

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for chop filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate KAMA (10-period ER) on close prices
    close = prices['close'].values
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0])).rolling(window=10, min_periods=1).sum()
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (0.66 - 0.06) + 0.06) ** 2
    kama = np.full_like(close, np.nan, dtype=float)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h - low_12h)
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR(14)
    atr_12h = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    range_14 = highest_high - lowest_low
    chop = np.where(range_14 != 0, 100 * np.log10(sum_tr14 / range_14) / np.log10(14), 50)
    
    # Align KAMA, RSI, Chop to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, prices, kama)  # already LTF
    rsi_aligned = align_htf_to_ltf(prices, prices, rsi.values)  # already LTF
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    signals = np.zeros(n)
    
    for i in range(14, n):
        # Skip if data not ready
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i]):
            continue
        
        # KAMA direction: rising if current > previous
        kama_rising = kama_aligned[i] > kama_aligned[i-1]
        kama_falling = kama_aligned[i] < kama_aligned[i-1]
        
        # Chop filter: choppy market (mean-reversion prone)
        choppy = chop_aligned[i] > 61.8
        
        if choppy:
            # Long: KAMA rising and RSI > 50
            if kama_rising and rsi_aligned[i] > 50:
                signals[i] = 0.25
            # Short: KAMA falling and RSI < 50
            elif kama_falling and rsi_aligned[i] < 50:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:
            # Not choppy - stay flat to avoid trend whipsaws
            signals[i] = 0.0
    
    return signals

name = "4h_KAMA10_RSI14_12hChop618"
timeframe = "4h"
leverage = 1.0