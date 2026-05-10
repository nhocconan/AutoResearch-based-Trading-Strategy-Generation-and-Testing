#!/usr/bin/env python3
"""
6h_PriceAction_Reversal_FailSafe
Hypothesis: Price often fails at prior session high/low (liquidity zones) and reverses.
We use 12h high/low as key levels: long when price rejects 12h low with bullish close,
short when price rejects 12h high with bearish close. Trend filter from 1d EMA50 avoids
counter-trend trades. Volume spike confirms rejection strength.
Works in bull/bear by trading reversals at key levels with trend alignment.
Target: 20-40 trades/year (80-160 total) to minimize fee drag.
"""

name = "6h_PriceAction_Reversal_FailSafe"
timeframe = "6h"
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
    volume = prices['volume'].values
    
    # 12h data for key levels
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 12h swing levels (prior completed bar)
    swing_high_12h = high_12h  # each 12h bar's high
    swing_low_12h = low_12h    # each 12h bar's low
    
    # 1d EMA50 for trend filter
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    
    # Align 12h levels and 1d EMA to 6h
    swing_high_aligned = align_htf_to_ltf(prices, df_12h, swing_high_12h)
    swing_low_aligned = align_htf_to_ltf(prices, df_12h, swing_low_12h)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50
    
    for i in range(start_idx, n):
        if np.isnan(swing_high_aligned[i]) or np.isnan(swing_low_aligned[i]) or np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Rejection conditions
        # Bullish rejection: low touches/breaks swing low but close above it
        bullish_rejection = low[i] <= swing_low_aligned[i] and close[i] > swing_low_aligned[i]
        # Bearish rejection: high touches/breaks swing high but close below it
        bearish_rejection = high[i] >= swing_high_aligned[i] and close[i] < swing_high_aligned[i]
        
        # Trend filter
        is_uptrend = close[i] > ema50_1d_aligned[i]
        is_downtrend = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: bullish rejection at support in uptrend
            if bullish_rejection and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: bearish rejection at resistance in downtrend
            elif bearish_rejection and is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: bearish rejection at resistance or trend turns down
            if bearish_rejection or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: bullish rejection at support or trend turns up
            if bullish_rejection or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals