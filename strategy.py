#!/usr/bin/env python3
"""
1d Bollinger Breakout with Weekly Trend Filter and Volume Confirmation v1
Hypothesis: Bollinger Band breakouts capture momentum bursts; weekly EMA filter ensures
trading in the direction of the higher timeframe trend; volume confirms breakout strength.
Designed for 75-150 trades over 4 years to minimize fee drag while adapting to bull/bear
markets via weekly trend filter. Works in both regimes: breakouts work in trending
markets, and the trend filter prevents counter-trend trades in choppy/ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_bollinger_breakout_weekly_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA(50) for trend filter
    close_weekly = df_weekly['close'].values
    ema_weekly = pd.Series(close_weekly).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Daily data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = ma + bb_std * std
    lower = ma - bb_std * std
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(bb_period, 20)  # Bollinger and volume
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ma[i]) or np.isnan(std[i]) or np.isnan(ema_weekly_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite Bollinger breakout or weekly trend reversal
        if position == 1:  # long position
            # Exit: price closes below middle band OR weekly trend turns down
            if close[i] < ma[i] or close[i] < ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above middle band OR weekly trend turns up
            if close[i] > ma[i] or close[i] > ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Bollinger breakout + weekly trend + volume
            bull_breakout = close[i] > upper[i] and close[i] > ema_weekly_aligned[i]
            bear_breakout = close[i] < lower[i] and close[i] < ema_weekly_aligned[i]
            volume_filter = volume[i] > vol_ma[i] * 1.5
            
            if bull_breakout and volume_filter:
                signals[i] = 0.25
                position = 1
            elif bear_breakout and volume_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals