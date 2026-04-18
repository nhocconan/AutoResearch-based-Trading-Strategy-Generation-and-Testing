#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray system with 1d trend filter. Elder Ray uses bull power (high-EMA13) and bear power (low-EMA13) to measure buying/selling pressure.
In bull markets: buy when bull power > 0 and rising, sell when bull power turns negative.
In bear markets: sell when bear power < 0 and falling, cover when bear power turns positive.
The 1d EMA50 filter ensures we only trade in the direction of the higher timeframe trend.
Designed for 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = calculate_ema(close_1d, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate EMA13 for Elder Ray (6-period EMA on 6h data)
    ema13 = calculate_ema(close, 13)
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # High minus EMA13
    bear_power = low - ema13   # Low minus EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # need EMA13 calculation
    
    for i in range(start_idx, n):
        # Skip if trend filter not available
        if np.isnan(ema50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1d EMA50
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Look for entries
            if uptrend:
                # In uptrend, look for bullish momentum
                if bull_power[i] > 0 and (i == start_idx or bull_power[i] > bull_power[i-1]):
                    signals[i] = 0.25
                    position = 1
            elif downtrend:
                # In downtrend, look for bearish momentum
                if bear_power[i] < 0 and (i == start_idx or bear_power[i] < bear_power[i-1]):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: bull power turns negative or trend changes
            if bull_power[i] <= 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bear power turns positive or trend changes
            if bear_power[i] >= 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dEMA50_TrendFilter"
timeframe = "6h"
leverage = 1.0