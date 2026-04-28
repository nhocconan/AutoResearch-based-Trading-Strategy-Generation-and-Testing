#!/usr/bin/env python3
"""
4h_PriceAction_RangeBreakout_12hTrend_1dVolatility
Hypothesis: Combines 12h EMA trend filter with daily ATR-based volatility filter and 4h price action breakouts.
Designed for low trade frequency (<25/year) to minimize fee burn while capturing strong directional moves 
in both bull and bear markets by requiring alignment across multiple timeframes and volatility confirmation.
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
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Get 1d data for ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 14-period ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 4-period ATR for breakout threshold
    tr1_4h = high - low
    tr2_4h = np.abs(high - np.roll(close, 1))
    tr3_4h = np.abs(low - np.roll(close, 1))
    tr1_4h[0] = 0
    tr2_4h[0] = 0
    tr3_4h[0] = 0
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_4 = pd.Series(tr_4h).rolling(window=4, min_periods=4).mean().values
    
    # Calculate 20-period high/low for breakout levels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            np.isnan(atr_4[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Volatility filter: current ATR > average ATR (avoid low volatility periods)
        vol_filter = atr_4[i] > atr_14_aligned[i] * 0.5
        
        # Breakout conditions: price breaks 20-period high/low with trend and volatility confirmation
        long_breakout = close[i] > highest_20[i]
        short_breakout = close[i] < lowest_20[i]
        
        long_entry = long_breakout and vol_filter and uptrend
        short_entry = short_breakout and vol_filter and downtrend
        
        # Exit conditions: price returns to midpoint of 20-period range or trend reverses
        midpoint_20 = (highest_20[i] + lowest_20[i]) / 2
        long_exit = close[i] < midpoint_20
        short_exit = close[i] > midpoint_20
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_PriceAction_RangeBreakout_12hTrend_1dVolatility"
timeframe = "4h"
leverage = 1.0