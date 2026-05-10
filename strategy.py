#!/usr/bin/env python3
# 1D_WickReversal_LiquiditySweep
# Hypothesis: Trade reversals after liquidity sweeps (wick closes beyond recent high/low but price closes back within range) on daily timeframe.
# Long when: price makes new 20-day low but closes above prior day's low (bullish rejection).
# Short when: price makes new 20-day high but closes below prior day's high (bearish rejection).
# Uses weekly trend filter: only trade in direction of weekly EMA50 trend.
# Works in bull/bear by following higher timeframe trend and using price action rejection for entry.
# Target: 15-25 trades/year per symbol.

name = "1D_WickReversal_LiquiditySweep"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily range and wick detection
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Wick reversal conditions
    bullish_wick = (low <= lowest_low) & (close > low)  # New low but closed off low
    bearish_wick = (high >= highest_high) & (close < high)  # New high but closed off high
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > ema50_1w
    weekly_downtrend = close_1w < ema50_1w
    
    # Align weekly trend to daily
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = lookback
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        weekly_up = weekly_uptrend_aligned[i] > 0.5
        weekly_down = weekly_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: weekly uptrend + bullish wick rejection
            if weekly_up and bullish_wick[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly downtrend + bearish wick rejection
            elif weekly_down and bearish_wick[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit conditions: trend reverses or opposite wick forms
            if not weekly_up or bearish_wick[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: trend reverses or opposite wick forms
            if not weekly_down or bullish_wick[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals