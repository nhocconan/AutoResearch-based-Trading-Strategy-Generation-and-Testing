#!/usr/bin/env python3
"""
4h_1d_kama_rsi_chop_v1
Strategy: 4h KAMA trend with RSI momentum and Choppiness Index regime filter
Timeframe: 4h
Leverage: 1.0
Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) to identify trend direction on 4h, combined with RSI for momentum confirmation and Choppiness Index to filter ranging markets. Enters long when KAMA slopes up, RSI > 50, and CHOP > 61.8 (ranging). Enters short when KAMA slopes down, RSI < 50, and CHOP > 61.8. Avoids trending markets (CHOP < 38.2) to reduce whipsaw. Designed to work in both bull and bear markets by focusing on mean-reversion in ranging conditions, which occur frequently in crypto.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_kama_rsi_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h KAMA (10, 2, 30)
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # needs correction
    # Correct volatility calculation: sum of absolute changes over period
    volatility = pd.Series(close).rolling(window=10, min_periods=10).apply(
        lambda x: np.sum(np.abs(np.diff(x))), raw=True
    ).values
    ER = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (ER * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1d RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 4h Choppiness Index (14)
    atr = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / (hh14 - ll14)) / np.log10(14)
    # Handle division by zero or invalid
    chop = np.where((hh14 - ll14) != 0, chop, 50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(kama[i-1]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # KAMA slope: rising if current > previous
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # RSI condition
        rsi = rsi_1d_aligned[i]
        rsi_bull = rsi > 50
        rsi_bear = rsi < 50
        
        # Choppiness Index: > 61.8 = ranging (good for mean reversion), < 38.2 = trending (avoid)
        chop_val = chop[i]
        ranging = chop_val > 61.8
        trending = chop_val < 38.2
        
        # Long: KAMA rising, RSI > 50, ranging market
        long_signal = kama_rising and rsi_bull and ranging
        
        # Short: KAMA falling, RSI < 50, ranging market
        short_signal = kama_falling and rsi_bear and ranging
        
        # Exit when KAMA changes direction or market becomes trending
        exit_long = position == 1 and (not kama_rising or not ranging)
        exit_short = position == -1 and (not kama_falling or not ranging)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals