#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily trend following with weekly filter.
# Long when daily close > daily EMA20 and weekly close > weekly EMA40.
# Short when daily close < daily EMA20 and weekly close < weekly EMA40.
# Exit when price crosses back over EMA20.
# Uses only 2 EMA filters to minimize overtrading.
# Target: 15-25 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for EMA20
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Load weekly data for EMA40
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    
    # Align to daily timeframe
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if NaN in critical values
        if np.isnan(ema20_1d_aligned[i]) or np.isnan(ema40_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema20 = ema20_1d_aligned[i]
        ema40 = ema40_1w_aligned[i]
        
        if position == 0:
            # Long: price above both EMAs
            if price > ema20 and ema20 > ema40:
                signals[i] = 0.25
                position = 1
            # Short: price below both EMAs
            elif price < ema20 and ema20 < ema40:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below EMA20
            if price < ema20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA20
            if price > ema20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_EMA20_EMA40_Trend_Filter_v1"
timeframe = "1d"
leverage = 1.0