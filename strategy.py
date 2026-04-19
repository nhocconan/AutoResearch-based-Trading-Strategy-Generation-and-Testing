#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_Turtle_Soup_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # 1d ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr_1d = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)), np.absolute(low_1d - np.roll(close_1d, 1)))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 1w pivot levels for directional bias
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    prev_high_1w = np.concatenate([[np.nan], high_1w[:-1]])
    prev_low_1w = np.concatenate([[np.nan], low_1w[:-1]])
    prev_close_1w = np.concatenate([[np.nan], close_1w[:-1]])
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    r1_1w = 2 * pivot_1w - prev_low_1w
    s1_1w = 2 * pivot_1w - prev_high_1w
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # 4h Donchian channels (20-period)
    donch_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h ATR for position sizing and stops
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if np.isnan(atr_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or \
           np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or \
           np.isnan(s1_1w_aligned[i]) or np.isnan(donch_high_20[i]) or np.isnan(donch_low_20[i]) or np.isnan(atr_4h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_4h[i]
        
        # Entry conditions: Turtle Soup pattern (false breakout reversal)
        # Long: price breaks below 20-period low then reverses back above it
        # Short: price breaks above 20-period high then reverses back below it
        
        # Check for false breakdown (long setup)
        false_breakdown = (low[i-1] < donch_low_20[i-1]) and (close[i] > donch_low_20[i-1])
        # Check for false breakout (short setup)
        false_breakout = (high[i-1] > donch_high_20[i-1]) and (close[i] < donch_high_20[i-1])
        
        # Additional filters: trend bias and weekly pivot alignment
        long_bias = price > ema200_1d_aligned[i]
        short_bias = price < ema200_1d_aligned[i]
        pw_long_bias = price > pivot_1w_aligned[i]
        pw_short_bias = price < pivot_1w_aligned[i]
        
        if position == 0:
            # Long: false breakdown + long bias + above weekly pivot
            if false_breakdown and long_bias and pw_long_bias:
                signals[i] = 0.25
                position = 1
            # Short: false breakout + short bias + below weekly pivot
            elif false_breakout and short_bias and pw_short_bias:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price reaches 20-period high or stops below entry
            if price >= donch_high_20[i] or price < close[i-1]:  # Simple trailing stop
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price reaches 20-period low or stops above entry
            if price <= donch_low_20[i] or price > close[i-1]:  # Simple trailing stop
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals