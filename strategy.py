#!/usr/bin/env python3
"""
12h_1wCVDivergence_Trend_Breakout
Hypothesis: Breakouts above weekly highs/lows with on-balance volume divergence confirmation and weekly trend filter.
In bull markets, buy breakouts above weekly highs with rising OBV; in bear markets, sell breakdowns below weekly lows with falling OBV.
Weekly trend filter ensures alignment with higher timeframe momentum. Volume divergence filters false breakouts.
Works in both bull and bear markets by using OBV divergence to confirm institutional participation.
"""

name = "12h_1wCVDivergence_Trend_Breakout"
timeframe = "12h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # === Weekly OHLC for Breakout Levels ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ph_w = df_1w['high'].values
    pl_w = df_1w['low'].values
    
    # Align weekly high/low to 12h timeframe
    weekly_high = align_htf_to_ltf(prices, df_1w, ph_w)
    weekly_low = align_htf_to_ltf(prices, df_1w, pl_w)
    
    # === Weekly Trend Filter (EMA50) ===
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === On-Balance Volume (OBV) for Divergence ===
    # Calculate OBV on 12h data
    obv = np.zeros(n)
    obv[0] = volume[0]
    for i in range(1, n):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    
    # OBV EMA for divergence detection
    obv_ema20 = pd.Series(obv).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(weekly_high[i]) or np.isnan(weekly_low[i]) or 
            np.isnan(ema50_12h[i]) or np.isnan(obv_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price above weekly high with bullish OBV divergence and uptrend
            if (close[i] > weekly_high[i] and 
                obv[i] > obv_ema20[i] and 
                close[i] > ema50_12h[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price below weekly low with bearish OBV divergence and downtrend
            elif (close[i] < weekly_low[i] and 
                  obv[i] < obv_ema20[i] and 
                  close[i] < ema50_12h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below weekly low or OBV turns bearish
            if close[i] < weekly_low[i] or obv[i] < obv_ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price breaks above weekly high or OBV turns bullish
            if close[i] > weekly_high[i] or obv[i] > obv_ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals