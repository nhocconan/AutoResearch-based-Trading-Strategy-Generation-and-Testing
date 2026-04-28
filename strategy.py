#!/usr/bin/env python3
"""
1d_KAMA_Slope_Trend_WeakLong_Short
Hypothesis: On daily timeframe, enter long when KAMA slope turns positive and short when KAMA slope turns negative, with weak position sizing (0.25 long, -0.15 short) to profit from trends while limiting drawdown in sideways/choppy markets. Uses weekly trend filter (price above/below weekly EMA50) to avoid counter-trend trades. Designed for low trade frequency (~10-25/year) to minimize fee decay. Targets BTC/ETH primarily with proven KAMA trend-following edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly 50 EMA for trend filter
    close_weekly = df_weekly['close'].values
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA to daily timeframe
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Weekly trend: bullish when price > EMA50, bearish when price < EMA50
    weekly_uptrend = close > ema50_weekly_aligned
    weekly_downtrend = close < ema50_weekly_aligned
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    # Parameters: ER length = 10, Fast EMA = 2, Slow EMA = 30
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None
    
    # Proper ER calculation
    er = np.zeros(n)
    for i in range(n):
        if i < 10:
            er[i] = np.nan
        else:
            direction = np.abs(close[i] - close[i-10])
            volatility_sum = np.sum(np.abs(np.diff(close[i-9:i+1])))  # 10-period volatility
            er[i] = direction / volatility_sum if volatility_sum != 0 else 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)      # EMA(2)
    slow_sc = 2 / (30 + 1)     # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA slope (1-period difference)
    kama_slope = np.diff(kama, prepend=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_weekly_aligned[i]) or np.isnan(kama_slope[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with weekly trend alignment
        long_entry = kama_slope[i] > 0 and weekly_uptrend[i]
        short_entry = kama_slope[i] < 0 and weekly_downtrend[i]
        
        # Exit when slope reverses
        long_exit = kama_slope[i] < 0
        short_exit = kama_slope[i] > 0
        
        if long_entry and position <= 0:
            signals[i] = 0.25   # Weak long
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.15  # Weak short
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to weak short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to weak long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.15
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_KAMA_Slope_Trend_WeakLong_Short"
timeframe = "1d"
leverage = 1.0