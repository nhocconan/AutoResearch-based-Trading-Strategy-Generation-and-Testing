#!/usr/bin/env python3
"""
1d_1w_TurtleTrend_Strategy
Hypothesis: Turtle Trading breakout system on daily timeframe with weekly trend filter.
Buy breakouts above 20-day high in weekly uptrend, sell breakdowns below 20-day low in weekly downtrend.
Works in bull markets (breakouts) and bear markets (breakdowns). Low frequency (~10-20/year) minimizes fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_TurtleTrend_Strategy"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DONCHIAN CHANNEL (20-period) ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === WEEKLY TREND FILTER (50-period EMA) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === VOLUME CONFIRMATION (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long breakout: price above 20-day high + volume + weekly uptrend
        long_breakout = (close[i] > high_roll[i]) and (vol_ratio[i] > 1.5) and (close[i] > ema50_1w_aligned[i])
        
        # Short breakdown: price below 20-day low + volume + weekly downtrend
        short_breakdown = (close[i] < low_roll[i]) and (vol_ratio[i] > 1.5) and (close[i] < ema50_1w_aligned[i])
        
        # Exit when price returns to 10-day mean (mean reversion)
        exit_long = close[i] < pd.Series(close).rolling(window=10, min_periods=10).mean().iloc[i] and position == 1
        exit_short = close[i] > pd.Series(close).rolling(window=10, min_periods=10).mean().iloc[i] and position == -1
        
        # Execute trades
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakdown and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals