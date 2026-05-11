#!/usr/bin/env python3
"""
12h_WB_Signal_1wTrend_Filtered
Hypothesis: On 12h timeframe, enter long when price breaks above the weekly Bollinger Upper Band (20, 2) and weekly ADX > 25, exit when price closes below the weekly Bollinger Middle Band (20). Reverse for shorts. Weekly trend filter ensures alignment with higher timeframe momentum. Bollinger Bands provide volatility-based breakout levels. Designed for low trade frequency (<30/year) to avoid fee drag while capturing strong trending moves. Works in bull markets via long breakouts and in bear markets via short breakdowns.
"""

name = "12h_WB_Signal_1wTrend_Filtered"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for Bollinger Bands and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly Bollinger Bands (20, 2) ---
    close_1w = df_1w['close'].values
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper = sma_20 + 2 * std_20
    middle = sma_20
    lower = sma_20 - 2 * std_20
    
    # Align BB to 12h
    upper_12h = align_htf_to_ltf(prices, df_1w, upper)
    middle_12h = align_htf_to_ltf(prices, df_1w, middle)
    lower_12h = align_htf_to_ltf(prices, df_1w, lower)
    
    # --- Weekly ADX (14) for trend strength ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.diff(high_1w, prepend=high_1w[0])
    down_move = -np.diff(low_1w, prepend=low_1w[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1w
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1w
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_12h = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40  # for BB and ADX
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(upper_12h[i]) or np.isnan(middle_12h[i]) or 
            np.isnan(adx_12h[i])):
            if position != 0:
                signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Weekly trend filter: only trade in strong trends
        strong_trend = adx_12h[i] > 25
        
        if position == 0 and strong_trend:
            # Look for breakouts in direction of weekly trend
            if close[i] > upper_12h[i]:
                signals[i] = 0.25  # long breakout
                position = 1
            elif close[i] < lower_12h[i]:
                signals[i] = -0.25  # short breakdown
                position = -1
        elif position == 1:
            # Long position: exit when price closes below weekly middle band
            if close[i] < middle_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above weekly middle band
            if close[i] > middle_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals