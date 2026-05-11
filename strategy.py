#!/usr/bin/env python3
"""
4h_Chandelier_Exit_System_v1
Hypothesis: Uses Chandelier Exit for trend following with volatility-based stops.
In bull markets, captures trends with wide stops; in bear markets, avoids whipsaws
by using ATR-based exits that adapt to volatility. Combines with 1-week trend filter
to avoid counter-trend trades. Target: 20-50 trades over 4 years (5-12/year).
"""

name = "4h_Chandelier_Exit_System_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === 1W Data for Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === Chandelier Exit Components ===
    # ATR(22) for volatility
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=22, adjust=False, min_periods=22).mean().values
    
    # 22-period highest high and lowest low
    highest_high = pd.Series(high).rolling(window=22, min_periods=22).max().values
    lowest_low = pd.Series(low).rolling(window=22, min_periods=22).min().values
    
    # Chandelier Exit: Long exit = highest high - 3*ATR, Short exit = lowest low + 3*ATR
    chandelier_long_exit = highest_high - 3.0 * atr
    chandelier_short_exit = lowest_low + 3.0 * atr
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(chandelier_long_exit[i]) or 
            np.isnan(chandelier_short_exit[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above Chandelier long exit AND weekly uptrend
            if close[i] > chandelier_long_exit[i] and ema50_1w_aligned[i] < close[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below Chandelier short exit AND weekly downtrend
            elif close[i] < chandelier_short_exit[i] and ema50_1w_aligned[i] > close[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below Chandelier long exit
            if close[i] < chandelier_long_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above Chandelier short exit
            if close[i] > chandelier_short_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals