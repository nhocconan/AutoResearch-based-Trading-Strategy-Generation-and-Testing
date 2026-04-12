#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d_1w_camarilla_pivot_reversal
# Uses weekly Camarilla pivot levels on daily chart for mean reversion trades.
# Long when price touches or crosses below L3 (75% of prior week range) with rejection (close > open).
# Short when price touches or crosses above H3 (125% of prior week range) with rejection (close < open).
# Exits when price reaches the opposite H3/L3 level or weekly pivot point.
# Designed for low trade frequency (target: 10-30 trades/year) to minimize fee drag.
# Works in ranging markets via mean reversion at extreme weekly levels.
# Works in trending markets via rejection at weekly support/resistance.
# Focus on BTC/ETH as primary targets.

name = "1d_1w_camarilla_pivot_reversal"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 10:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_price = prices['open'].values
    
    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (based on prior week's OHLC)
    # H4 = High + 1.5 * (High - Low)
    # H3 = High + 1.0 * (High - Low)
    # L3 = Low - 1.0 * (High - Low)
    # L4 = Low - 1.5 * (High - Low)
    # Pivot = (High + Low + Close) / 3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly range
    weekly_range = high_1w - low_1w
    
    # Camarilla levels
    h4 = high_1w + 1.5 * weekly_range
    h3 = high_1w + 1.0 * weekly_range
    l3 = low_1w - 1.0 * weekly_range
    l4 = low_1w - 1.5 * weekly_range
    pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Align weekly Camarilla levels to daily timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1w, h4)
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1w, l4)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # start after first bar to have prior weekly data
        # Skip if data not ready
        if np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(pivot_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Long signal: price touches/crosses below L3 with bullish rejection (close > open)
        if low[i] <= l3_aligned[i] and close[i] > open_price[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price touches/crosses above H3 with bearish rejection (close < open)
        elif high[i] >= h3_aligned[i] and close[i] < open_price[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price reaches opposite H3/L3 level or weekly pivot
        elif position == 1 and (high[i] >= h3_aligned[i] or close[i] >= pivot_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (low[i] <= l3_aligned[i] or close[i] <= pivot_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals