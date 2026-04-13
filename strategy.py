#!/usr/bin/env python3
"""
1d_1w_KAMA_Trend_Filter_With_Volume_Confirmation
Hypothesis: KAMA adapts to market noise, providing reliable trend direction on daily.
Trend confirmed by price > KAMA (long) or < KAMA (short) with volume expansion.
Uses weekly trend filter: only trade long when weekly close > weekly EMA20, short when <.
Volume confirmation reduces false signals. Works in bull (trend following) and bear (counter-trend reversals via weekly filter).
Target: 10-20 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for KAMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA (20, 2, 30)
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * 0.6 + 0.06) ** 2  # 2 = fast SC, 30 = slow SC
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: current volume > 1.3x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price > KAMA, weekly close > weekly EMA20, volume expansion
        long_condition = (close[i] > kama_aligned[i]) and \
                         (close_1w[-1] > ema_20_1w_aligned[i] if i < len(ema_20_1w_aligned) else False) and \
                         volume_expansion[i]
        
        # Short conditions: price < KAMA, weekly close < weekly EMA20, volume expansion
        short_condition = (close[i] < kama_aligned[i]) and \
                          (close_1w[-1] < ema_20_1w_aligned[i] if i < len(ema_20_1w_aligned) else False) and \
                          volume_expansion[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1d_1w_KAMA_Trend_Filter_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0