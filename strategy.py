#!/usr/bin/env python3
"""
1D_WEEKLY_NATR_BREAKOUT_1W_TREND_FILTER
Hypothesis: Weekly ATR-based volatility breakout with trend filter captures strong moves in both bull and bear markets.
Long when price breaks above weekly high + 0.5*weekly ATR with weekly EMA50 uptrend.
Short when price breaks below weekly low - 0.5*weekly ATR with weekly EMA50 downtrend.
Uses daily timeframe for entries with weekly volatility and trend as structure. Target: 10-25 trades/year on 1d timeframe.
"""
name = "1D_WEEKLY_NATR_BREAKOUT_1W_TREND_FILTER"
timeframe = "1d"
leverage = 1.0

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
    
    # Weekly data for volatility and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly ATR(10) for volatility measurement
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Weekly EMA50 for trend filter
    ema50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Weekly high/low for breakout levels
    weekly_high = pd.Series(high_1w).rolling(window=1, min_periods=1).max().values  # current week high
    weekly_low = pd.Series(low_1w).rolling(window=1, min_periods=1).min().values    # current week low
    
    # Align weekly data to daily timeframe (wait for weekly bar close)
    atr10_aligned = align_htf_to_ltf(prices, df_1w, atr10)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50)
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(atr10_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        atr_val = atr10_aligned[i]
        if np.isnan(atr_val) or atr_val <= 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above weekly high + 0.5*weekly ATR in uptrend
            if (close[i] > weekly_high_aligned[i] + 0.5 * atr_val and 
                close[i] > ema50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly low - 0.5*weekly ATR in downtrend
            elif (close[i] < weekly_low_aligned[i] - 0.5 * atr_val and 
                  close[i] < ema50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below weekly low or trend reversal
            if (close[i] < weekly_low_aligned[i] or 
                close[i] < ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above weekly high or trend reversal
            if (close[i] > weekly_high_aligned[i] or 
                close[i] > ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals