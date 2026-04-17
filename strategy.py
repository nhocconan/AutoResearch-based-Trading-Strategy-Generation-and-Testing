#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Trend_Reversal_v1
Uses daily Camarilla pivot levels (S1, S2, R1, R2) on 12h timeframe.
Long: price touches S1/S2 with bullish reversal (close > open) and price > 1w EMA200.
Short: price touches R1/R2 with bearish reversal (close < open) and price < 1w EMA200.
Exit: price reaches opposite S/R level or EMA200 cross.
Designed to capture reversals at key levels with trend filter.
Target: 50-150 total trades over 4 years (12-37/year).
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
    open_price = prices['open'].values
    
    # === 1d High/Low/CLOSE for Camarilla pivots ===
    df_1d = get_htf_data(prices, '1d')
    # Ensure we have enough data
    if len(df_1d) < 2:
        return np.zeros(n)
    # Use previous day's OHLC for today's pivots (standard)
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    # Shift by 1 to use previous day's data
    prev_high = np.concatenate([[prev_high[0]], prev_high[:-1]])
    prev_low = np.concatenate([[prev_low[0]], prev_low[:-1]])
    prev_close = np.concatenate([[prev_close[0]], prev_close[:-1]])
    
    # Camarilla levels
    range_val = prev_high - prev_low
    S1 = prev_close - (range_val * 1.1 / 12)
    S2 = prev_close - (range_val * 1.1 / 6)
    R1 = prev_close + (range_val * 1.1 / 12)
    R2 = prev_close + (range_val * 1.1 / 6)
    
    # Align pivots to 12h timeframe
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    
    # === 1w EMA200 for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(S1_aligned[i]) or np.isnan(S2_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(R2_aligned[i]) or
            np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price touches S1/S2, bullish candle, above 1w EMA200
            touch_s1 = low[i] <= S1_aligned[i] and close[i] > S1_aligned[i]
            touch_s2 = low[i] <= S2_aligned[i] and close[i] > S2_aligned[i]
            bullish = close[i] > open_price[i]
            above_ema = close[i] > ema_200_1w_aligned[i]
            
            if (touch_s1 or touch_s2) and bullish and above_ema:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price touches R1/R2, bearish candle, below 1w EMA200
            touch_r1 = high[i] >= R1_aligned[i] and close[i] < R1_aligned[i]
            touch_r2 = high[i] >= R2_aligned[i] and close[i] < R2_aligned[i]
            bearish = close[i] < open_price[i]
            below_ema = close[i] < ema_200_1w_aligned[i]
            
            if (touch_r1 or touch_r2) and bearish and below_ema:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price reaches R1/R2 or closes below EMA200
            if (high[i] >= R1_aligned[i] or 
                high[i] >= R2_aligned[i] or
                close[i] < ema_200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches S1/S2 or closes above EMA200
            if (low[i] <= S1_aligned[i] or 
                low[i] <= S2_aligned[i] or
                close[i] > ema_200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_Trend_Reversal_v1"
timeframe = "12h"
leverage = 1.0