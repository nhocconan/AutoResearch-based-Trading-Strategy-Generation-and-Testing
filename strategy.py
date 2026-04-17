#!/usr/bin/env python3
"""
12h_KAMA_Trend_RSI_Filter_v1
KAMA(14) trend direction + RSI(14) pullback entry on 12h timeframe.
Trend filter: price above/below 1d EMA50.
Exit when KAMA reverses or RSI overextends.
Designed to capture trend continuation with pullback entries in both bull and bear markets.
Target: 50-150 total trades over 4 years (12-37/year).
"""

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
    
    # === KAMA(14) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=14))
    abs_change = np.abs(np.diff(close, n=1))
    er = np.zeros_like(close)
    er[14:] = change[14:] / (np.abs(np.diff(close, n=1))[13:].cumsum() - np.abs(np.diff(close, n=1))[13:-14].cumsum() + 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.full_like(close, np.nan)
    kama[14] = close[14]
    for i in range(15, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI(14) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmrow = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmrow, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price above KAMA (uptrend), RSI < 40 (pullback), price above 1d EMA50
            if (close[i] > kama[i] and 
                rsi[i] < 40 and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price below KAMA (downtrend), RSI > 60 (pullback), price below 1d EMA50
            elif (close[i] < kama[i] and 
                  rsi[i] > 60 and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price below KAMA OR RSI > 70 (overbought)
            if (close[i] < kama[i] or 
                rsi[i] > 70):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above KAMA OR RSI < 30 (oversold)
            if (close[i] > kama[i] or 
                rsi[i] < 30):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Trend_RSI_Filter_v1"
timeframe = "12h"
leverage = 1.0