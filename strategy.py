#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Chandelier_Exit_Trend_Follow_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly ATR for trend direction
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr_1w = np.zeros_like(close_1w)
    atr_1w[0] = np.nan
    for i in range(1, len(tr) + 1):
        if i < 22:
            atr_1w[i] = np.nan
        else:
            atr_1w[i] = np.mean(tr[i-21:i])
    
    # Chandelier Exit calculation
    # Long exit: highest high since entry minus ATR*multiplier
    # Short exit: lowest low since entry plus ATR*multiplier
    # We use it as trend filter: price above long exit = uptrend, below short exit = downtrend
    atr_mult = 3.0
    chandelier_long = np.full_like(close_1w, np.nan)
    chandelier_short = np.full_like(close_1w, np.nan)
    
    highest_since = np.full_like(close_1w, np.nan)
    lowest_since = np.full_like(close_1w, np.nan)
    
    for i in range(1, len(close_1w)):
        if np.isnan(highest_since[i-1]):
            highest_since[i] = high_1w[i]
            lowest_since[i] = low_1w[i]
        else:
            highest_since[i] = max(highest_since[i-1], high_1w[i])
            lowest_since[i] = min(lowest_since[i-1], low_1w[i])
        
        chandelier_long[i] = highest_since[i] - atr_mult * atr_1w[i]
        chandelier_short[i] = lowest_since[i] + atr_mult * atr_1w[i]
    
    # Align Chandelier Exit levels to daily
    chandelier_long_d = align_htf_to_ltf(prices, df_1w, chandelier_long)
    chandelier_short_d = align_htf_to_ltf(prices, df_1w, chandelier_short)
    
    # Daily ATR for volatility filter
    tr1_d = high[1:] - low[1:]
    tr2_d = np.abs(high[1:] - close[:-1])
    tr3_d = np.abs(low[1:] - close[:-1])
    tr_d = np.maximum.reduce([tr1_d, tr2_d, tr3_d])
    atr_d = np.zeros_like(close)
    atr_d[0] = np.nan
    for i in range(1, len(tr_d) + 1):
        if i < 14:
            atr_d[i] = np.nan
        else:
            atr_d[i] = np.mean(tr_d[i-13:i])
    
    # Volatility filter: avoid extremely low volatility periods
    atr_ma_d = pd.Series(atr_d).rolling(window=20, min_periods=20).mean().values
    vol_filter = atr_d > 0.5 * atr_ma_d  # Avoid low volatility
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(chandelier_long_d[i]) or np.isnan(chandelier_short_d[i]) or np.isnan(atr_d[i]) or np.isnan(atr_ma_d[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter using Chandelier Exit
        # Long when price above long exit (uptrend)
        # Short when price below short exit (downtrend)
        if position == 0:
            # Enter long: price above Chandelier long exit and volatility filter
            if price > chandelier_long_d[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price below Chandelier short exit and volatility filter
            elif price < chandelier_short_d[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below Chandelier long exit (trend change)
            if price < chandelier_long_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above Chandelier short exit (trend change)
            if price > chandelier_short_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals