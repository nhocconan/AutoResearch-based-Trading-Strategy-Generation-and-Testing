#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_1d_ADX_Filter
Hypothesis: Use KAMA (Kaufman Adaptive Moving Average) on 4h for trend direction, filtered by 1d ADX for trend strength. Only trade when ADX > 25 (strong trend). Enter long when price > KAMA and ADX rising, short when price < KAMA and ADX rising. Exit when price crosses KAMA or ADX falls below 20. Uses 4h timeframe for entries, 1d for trend filter. Aims for 20-40 trades/year to minimize fee drag. Works in trending markets (both bull and bear) by capturing sustained moves.
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
    
    # Calculate KAMA on 4h
    def kama(close, period=10, fast=2, slow=30):
        # Efficiency ratio
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # Initialize KAMA
        kama = np.full_like(close, np.nan)
        kama[period] = close[period]
        for i in range(period+1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_val = kama(close, period=10, fast=2, slow=30)
    
    # Calculate ADX on 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def smooth(x, period):
        return pd.Series(x).ewm(alpha=1/period, adjust=False).mean().values
    
    atr = smooth(tr, 14)
    plus_di = 100 * smooth(plus_dm, 14) / atr
    minus_di = 100 * smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth(dx, 14)
    
    # Align ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for KAMA and ADX
    start_idx = max(30, 30)
    
    for i in range(start_idx, n):
        if np.isnan(kama_val[i]) or np.isnan(adx_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > KAMA and ADX > 25 and rising
            if close[i] > kama_val[i] and adx_aligned[i] > 25 and adx_aligned[i] > adx_aligned[i-1]:
                signals[i] = size
                position = 1
            # Short: price < KAMA and ADX > 25 and rising
            elif close[i] < kama_val[i] and adx_aligned[i] > 25 and adx_aligned[i] > adx_aligned[i-1]:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price < KAMA or ADX < 20
            if close[i] < kama_val[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price > KAMA or ADX < 20
            if close[i] > kama_val[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_KAMA_Trend_With_1d_ADX_Filter"
timeframe = "4h"
leverage = 1.0