#!/usr/bin/env python3
# 12h_1w_HighLowBreakout_TrendFilter
# Hypothesis: Breakouts above weekly high or below weekly low with trend confirmation (price above/below weekly EMA50) on 12h timeframe.
# Uses weekly high/low as dynamic support/resistance and EMA50 for trend filter to avoid false breakouts in ranging markets.
# Target: 20-50 trades per year per symbol to stay within optimal range for 12h timeframe.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_HighLowBreakout_TrendFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for high/low and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Calculate weekly high, low, and EMA50 ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 12h timeframe
    high_1w_aligned = align_htf_to_ltf(prices, df_1w, high_1w)
    low_1w_aligned = align_htf_to_ltf(prices, df_1w, low_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        high_val = high_1w_aligned[i]
        low_val = low_1w_aligned[i]
        ema50_val = ema50_1w_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(high_val) or np.isnan(low_val) or np.isnan(ema50_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly high AND above weekly EMA50 (uptrend)
            if close_val > high_val and close_val > ema50_val:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly low AND below weekly EMA50 (downtrend)
            elif close_val < low_val and close_val < ema50_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below weekly EMA50 (trend change)
            if close_val <= ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to or above weekly EMA50 (trend change)
            if close_val >= ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals