#!/usr/bin/env python3
"""
1d_KAMA_RSI_Chop_Filter_v1
Concept: Daily KAMA trend + RSI momentum + Choppiness regime filter.
- Long: KAMA rising AND RSI > 55 AND Chop > 61.8 (range)
- Short: KAMA falling AND RSI < 45 AND Chop > 61.8 (range)
- Exit: Opposite KAMA direction OR RSI crosses 50
- Position sizing: 0.25
- Target: 30-80 total trades over 4 years (7-20/year)
- Works in bear: range-bound markets favor mean reversion; chop filter avoids trending whipsaws
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_RSI_Chop_Filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Weekly: EMA Trend Filter (21) ===
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # === Daily: KAMA (ER=10) ===
    close = prices['close'].values
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === Daily: RSI (14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Daily: Choppiness Index (14) ===
    atr = np.zeros_like(close)
    tr1 = np.abs(np.subtract(prices['high'].values, prices['low'].values))
    tr2 = np.abs(np.subtract(prices['high'].values, np.roll(close, 1)))
    tr3 = np.abs(np.subtract(prices['low'].values, np.roll(close, 1)))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(prices['high'].values).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(prices['low'].values).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((np.sum(atr, axis=1) / (max_high - min_low))) / np.log10(14)
    chop = np.where((max_high - min_low) != 0, chop, 50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        kama_val = kama[i]
        kama_prev = kama[i-1]
        rsi_val = rsi[i]
        chop_val = chop[i]
        ema_21_1w = ema_21_1w_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(kama_val) or np.isnan(kama_prev) or np.isnan(rsi_val) or 
            np.isnan(chop_val) or np.isnan(ema_21_1w)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: weekly EMA direction
        kama_rising = kama_val > kama_prev
        kama_falling = kama_val < kama_prev
        
        if position == 0:
            # Long: KAMA rising AND RSI > 55 AND Chop > 61.8 (range)
            if kama_rising and rsi_val > 55 and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling AND RSI < 45 AND Chop > 61.8 (range)
            elif kama_falling and rsi_val < 45 and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA falling OR RSI crosses below 50
            if not kama_rising or rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA rising OR RSI crosses above 50
            if not kama_falling or rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals