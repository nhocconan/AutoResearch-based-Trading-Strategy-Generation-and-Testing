#!/usr/bin/env python3
"""
4h_1d_1w_Trend_Follow_With_Regime_Filter
Hypothesis: Use 1d EMA200 and 1w EMA40 to establish long-term trend direction, and enter on 4h pullbacks to the 20-period EMA when the 4h ADX indicates a trending market (ADX > 25). Exit when price crosses the 20 EMA in the opposite direction or ADX weakens (< 20). This captures trend continuation moves while avoiding choppy markets. Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend). Target: 20-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for long-term trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for super trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 1w EMA40 for super trend
    close_1w = df_1w['close'].values
    ema40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # Calculate 4h EMA20 for entry
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 4h ADX for trend strength
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - low[:-1]), np.absolute(low[1:] - high[:-1]))
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / (pd.Series(tr).ewm(alpha=1/14, adjust=False).mean() + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / (pd.Series(tr).ewm(alpha=1/14, adjust=False).mean() + 1e-10)
    dx = 100 * np.absolute(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(ema40_1w_aligned[i]) or 
            np.isnan(ema20[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Trend filters: 1d EMA200 and 1w EMA40 must agree
        uptrend = close[i] > ema200_1d_aligned[i] and ema200_1d_aligned[i] > ema40_1w_aligned[i]
        downtrend = close[i] < ema200_1d_aligned[i] and ema200_1d_aligned[i] < ema40_1w_aligned[i]
        
        # ADX trend strength filter
        trending = adx[i] > 25
        weak_trend = adx[i] < 20
        
        # Long: uptrend + ADX strong + price at EMA20 support
        long_condition = uptrend and trending and close[i] <= ema20[i] * 1.005  # within 0.5% of EMA20
        
        # Short: downtrend + ADX strong + price at EMA20 resistance
        short_condition = downtrend and trending and close[i] >= ema20[i] * 0.995  # within 0.5% of EMA20
        
        # Exit conditions: trend weakens or price crosses EMA20 in opposite direction
        exit_long = not uptrend or weak_trend or close[i] < ema20[i]
        exit_short = not downtrend or weak_trend or close[i] > ema20[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_1w_Trend_Follow_With_Regime_Filter"
timeframe = "4h"
leverage = 1.0