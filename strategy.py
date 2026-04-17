#!/usr/bin/env python3
"""
1d_KAMA_RSI_Chop_v1
KAMA(2,10) for trend direction + RSI(14) for momentum + Choppiness Index(14) for regime filter.
Long when KAMA rising, RSI>50, and chop>61.8 (range) for mean reversion to upside.
Short when KAMA falling, RSI<50, and chop>61.8 for mean reversion to downside.
Exit when chop<38.2 (trend) or RSI crosses 50 opposite.
Designed to work in both bull (trend following when chop low) and bear/range (mean reversion when chop high).
Target: 50-100 total trades over 4 years (12-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === KAMA(2,10) ===
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, 10))
    change[0:10] = 0  # first 10 values
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Calculate volatility using rolling sum of absolute changes
    vol_series = pd.Series(np.abs(np.diff(close), prepend=0))
    volatility = vol_series.rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppiness Index(14) ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    max_hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Choppy calculation
    chop = 100 * np.log10(tr_sum / (max_hh - min_ll)) / np.log10(14)
    # Handle division by zero when max_hh == min_ll
    chop = np.where((max_hh - min_ll) != 0, chop, 50)
    
    # === Weekly trend filter (1w) ===
    df_1w = get_htf_data(prices, '1w')
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Determine trend from weekly EMA200
        weekly_uptrend = close[i] > ema_200_1w_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long conditions: KAMA rising, RSI>50, chop>61.8 (range) for mean reversion
            # In strong trend (weekly uptrend), we can be more aggressive
            if (kama[i] > kama[i-1] and 
                rsi[i] > 50 and 
                chop[i] > 61.8):
                signals[i] = 0.25
                position = 1
                continue
            # Short conditions: KAMA falling, RSI<50, chop>61.8 (range) for mean reversion
            elif (kama[i] < kama[i-1] and 
                  rsi[i] < 50 and 
                  chop[i] > 61.8):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: chop<38.2 (trend) OR RSI<50 OR KAMA turns down
            if (chop[i] < 38.2 or 
                rsi[i] < 50 or 
                kama[i] < kama[i-1]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: chop<38.2 (trend) OR RSI>50 OR KAMA turns up
            if (chop[i] < 38.2 or 
                rsi[i] > 50 or 
                kama[i] > kama[i-1]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_Chop_v1"
timeframe = "1d"
leverage = 1.0