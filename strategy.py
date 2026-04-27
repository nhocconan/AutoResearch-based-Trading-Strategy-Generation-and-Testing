#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 12h Camarilla pivot levels for breakout entries with 4h RSI filter.
Camarilla levels provide institutional support/resistance; breakouts from these levels capture
institutional flow. RSI filter avoids overextended entries. Works in bull/bear by capturing
breakouts in trending phases. Target: 20-40 trades/year.
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
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 12h bar using previous bar's OHLC
    # Camarilla formula: 
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # H3 = Close + 1.125 * (High - Low)
    # L3 = Close - 1.125 * (High - Low)
    # H2 = Close + 0.75 * (High - Low)
    # L2 = Close - 0.75 * (High - Low)
    # H1 = Close + 0.5 * (High - Low)
    # L1 = Close - 0.5 * (High - Low)
    
    h4 = np.full(len(df_12h), np.nan)
    l4 = np.full(len(df_12h), np.nan)
    h3 = np.full(len(df_12h), np.nan)
    l3 = np.full(len(df_12h), np.nan)
    h2 = np.full(len(df_12h), np.nan)
    l2 = np.full(len(df_12h), np.nan)
    h1 = np.full(len(df_12h), np.nan)
    l1 = np.full(len(df_12h), np.nan)
    
    for i in range(1, len(df_12h)):
        high_prev = df_12h['high'].iloc[i-1]
        low_prev = df_12h['low'].iloc[i-1]
        close_prev = df_12h['close'].iloc[i-1]
        diff = high_prev - low_prev
        
        h4[i] = close_prev + 1.5 * diff
        l4[i] = close_prev - 1.5 * diff
        h3[i] = close_prev + 1.125 * diff
        l3[i] = close_prev - 1.125 * diff
        h2[i] = close_prev + 0.75 * diff
        l2[i] = close_prev - 0.75 * diff
        h1[i] = close_prev + 0.5 * diff
        l1[i] = close_prev - 0.5 * diff
    
    # Align Camarilla levels to 4h timeframe (wait for 12h bar close)
    h4_aligned = align_htf_to_ltf(prices, df_12h, h4)
    l4_aligned = align_htf_to_ltf(prices, df_12h, l4)
    h3_aligned = align_htf_to_ltf(prices, df_12h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_12h, l3)
    h2_aligned = align_htf_to_ltf(prices, df_12h, h2)
    l2_aligned = align_htf_to_ltf(prices, df_12h, l2)
    h1_aligned = align_htf_to_ltf(prices, df_12h, h1)
    l1_aligned = align_htf_to_ltf(prices, df_12h, l1)
    
    # Calculate 4h RSI(14) for momentum filter
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder smoothing for RSI
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    if n >= rsi_period:
        avg_gain[rsi_period-1] = np.mean(gain[1:rsi_period+1])
        avg_loss[rsi_period-1] = np.mean(loss[1:rsi_period+1])
        
        for i in range(rsi_period, n):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i]) / rsi_period
    
    rs = np.full(n, np.nan)
    rsi = np.full(n, 50.0)  # Default to neutral
    
    for i in range(rsi_period-1, n):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need Camarilla (1 bar) and RSI (14)
    start_idx = max(1, rsi_period-1)
    
    for i in range(start_idx, n):
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h2_aligned[i]) or np.isnan(l2_aligned[i]) or
            np.isnan(h1_aligned[i]) or np.isnan(l1_aligned[i]) or
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long entry: price breaks above H3 with RSI not overbought
            if price > h3_aligned[i] and rsi[i] < 70:
                signals[i] = size
                position = 1
            # Short entry: price breaks below L3 with RSI not oversold
            elif price < l3_aligned[i] and rsi[i] > 30:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below H1 or RSI overbought
            if price < h1_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price crosses above L1 or RSI oversold
            if price > l1_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_RSI14"
timeframe = "4h"
leverage = 1.0