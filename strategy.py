#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_Pivot_Breakout_With_TimeFilter
Hypothesis: Use 4h Camarilla pivot levels + 1d trend filter (price above/below 200 EMA) to determine direction, with 1h for precise entry timing and session filter (08-20 UTC) to reduce noise. Target: 15-37 trades/year per symbol.
"""

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
    
    # Calculate ATR for Camarilla levels
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[0], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=5, min_periods=5).mean().values
    
    # Calculate Camarilla levels from previous 4h bar
    def calculate_camarilla(high_val, low_val, close_val):
        range_val = high_val - low_val
        if range_val <= 0:
            return close_val, close_val, close_val, close_val
        multiplier = 1.1 / 12
        h4 = close_val + range_val * multiplier * 1.1
        h3 = close_val + range_val * multiplier * 1.5
        h2 = close_val + range_val * multiplier * 2.0
        h1 = close_val + range_val * multiplier * 2.6
        l1 = close_val - range_val * multiplier * 2.6
        l2 = close_val - range_val * multiplier * 2.0
        l3 = close_val - range_val * multiplier * 1.5
        l4 = close_val - range_val * multiplier * 1.1
        return h4, h3, h2, h1, l1, l2, l3, l4
    
    # Get 4h data once
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        h4_4h = h3_4h = h2_4h = h1_4h = l1_4h = l2_4h = l3_4h = l4_4h = np.full(len(prices), np.nan)
    else:
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        close_4h = df_4h['close'].values
        h4_4h_raw = np.full(len(close_4h), np.nan)
        h3_4h_raw = np.full(len(close_4h), np.nan)
        h2_4h_raw = np.full(len(close_4h), np.nan)
        h1_4h_raw = np.full(len(close_4h), np.nan)
        l1_4h_raw = np.full(len(close_4h), np.nan)
        l2_4h_raw = np.full(len(close_4h), np.nan)
        l3_4h_raw = np.full(len(close_4h), np.nan)
        l4_4h_raw = np.full(len(close_4h), np.nan)
        
        for i in range(1, len(close_4h)):
            h4, h3, h2, h1, l1, l2, l3, l4 = calculate_camarilla(high_4h[i-1], low_4h[i-1], close_4h[i-1])
            h4_4h_raw[i] = h4
            h3_4h_raw[i] = h3
            h2_4h_raw[i] = h2
            h1_4h_raw[i] = h1
            l1_4h_raw[i] = l1
            l2_4h_raw[i] = l2
            l3_4h_raw[i] = l3
            l4_4h_raw[i] = l4
        
        # Align to 1h timeframe
        h4_4h = align_htf_to_ltf(prices, df_4h, h4_4h_raw)
        h3_4h = align_htf_to_ltf(prices, df_4h, h3_4h_raw)
        h2_4h = align_htf_to_ltf(prices, df_4h, h2_4h_raw)
        h1_4h = align_htf_to_ltf(prices, df_4h, h1_4h_raw)
        l1_4h = align_htf_to_ltf(prices, df_4h, l1_4h_raw)
        l2_4h = align_htf_to_ltf(prices, df_4h, l2_4h_raw)
        l3_4h = align_htf_to_ltf(prices, df_4h, l3_4h_raw)
        l4_4h = align_htf_to_ltf(prices, df_4h, l4_4h_raw)
    
    # Get 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        ema200_1d = np.full(len(prices), np.nan)
    else:
        close_1d = df_1d['close'].values
        ema200_1d_raw = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
        ema200_1d = align_htf_to_ltf(prices, df_1d, ema200_1d_raw)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # 20% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(h4_4h[i]) or np.isnan(l4_4h[i]) or 
            np.isnan(ema200_1d[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Long: break above H4 with price above 1d EMA200
        long_signal = (high[i] > h4_4h[i] and close[i] > ema200_1d[i])
        # Short: break below L4 with price below 1d EMA200
        short_signal = (low[i] < l4_4h[i] and close[i] < ema200_1d[i])
        
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1h_4h_1d_Camarilla_Pivot_Breakout_With_TimeFilter"
timeframe = "1h"
leverage = 1.0