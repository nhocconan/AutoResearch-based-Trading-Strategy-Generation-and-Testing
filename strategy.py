#!/usr/bin/env python3
"""
4h_turtle_trading_v1
Hypothesis: 4h Turtle Trading system with Donchian breakouts, ATR-based position sizing, and trend filter.
Uses 20-period Donchian channels for breakouts and 25-period EMA for trend filtering.
Designed to work in both bull and bear markets by only trading in direction of higher timeframe trend.
Target: 20-40 trades/year (80-160 over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_turtle_trading_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h ATR(14) for volatility filtering
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(atr[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below 10-period low OR trend changes
            low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values[i]
            if close[i] < low_10 or ema_50_1d_aligned[i] < ema_50_1d_aligned[max(0, i-1)]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price crosses above 10-period high OR trend changes
            high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values[i]
            if close[i] > high_10 or ema_50_1d_aligned[i] > ema_50_1d_aligned[max(0, i-1)]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above 20-period high with uptrend
            if (not np.isnan(high_20[i]) and close[i] > high_20[i] and 
                ema_50_1d_aligned[i] > ema_50_1d_aligned[max(0, i-1)]):
                position = 1
                signals[i] = 0.30
            # Short entry: price breaks below 20-period low with downtrend
            elif (not np.isnan(low_20[i]) and close[i] < low_20[i] and 
                  ema_50_1d_aligned[i] < ema_50_1d_aligned[max(0, i-1)]):
                position = -1
                signals[i] = -0.30
    
    return signals