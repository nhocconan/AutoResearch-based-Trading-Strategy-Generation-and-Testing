#!/usr/bin/env python3
"""
1d_1w_KAMA_Trend_RSI
Hypothesis: Use weekly KAMA trend direction with daily RSI mean-reversion on 1d timeframe.
Long when weekly KAMA rising and daily RSI < 30; short when weekly KAMA falling and daily RSI > 70.
Exit when RSI reverts to neutral (40-60 range). Uses 0.25 position sizing.
Designed to capture trend-aligned mean-reversion moves in both bull and bear markets.
Target: 50-150 total trades over 4 years (12-37/year) on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_KAMA_Trend_RSI"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === WEEKLY KAMA TREND ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Efficiency Ratio for KAMA
    change = np.abs(np.diff(close_1w, n=9))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1w)), axis=1)  # 10-period volatility
    er = np.zeros_like(close_1w)
    er[9:] = change[9:] / volatility[9:]
    er[volatility == 0] = 0
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.full_like(close_1w, np.nan)
    kama[9] = close_1w[9]  # seed
    
    for i in range(10, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # KAMA direction (rising/falling)
    kama_rising = np.zeros_like(kama, dtype=bool)
    kama_falling = np.zeros_like(kama, dtype=bool)
    kama_rising[1:] = kama[1:] > kama[:-1]
    kama_falling[1:] = kama[1:] < kama[:-1]
    
    # Align to 1d timeframe
    kama_rising_1d = align_htf_to_ltf(prices, df_1w, kama_rising.astype(float))
    kama_falling_1d = align_htf_to_ltf(prices, df_1w, kama_falling.astype(float))
    
    # === DAILY RSI ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if not ready
        if (np.isnan(kama_rising_1d[i]) or np.isnan(kama_falling_1d[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: weekly KAMA rising and daily RSI oversold
        long_signal = (kama_rising_1d[i] > 0.5 and 
                      rsi[i] < 30)
        
        # Short: weekly KAMA falling and daily RSI overbought
        short_signal = (kama_falling_1d[i] > 0.5 and 
                       rsi[i] > 70)
        
        # Exit: RSI returns to neutral zone (40-60)
        exit_long = (position == 1 and 
                    rsi[i] >= 40)
        exit_short = (position == -1 and 
                     rsi[i] <= 60)
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals