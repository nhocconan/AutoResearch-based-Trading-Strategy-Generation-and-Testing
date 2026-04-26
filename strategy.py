#!/usr/bin/env python3
"""
6h_IBS_Reversion_12hTrend_Filter
Hypothesis: 6h mean reversion using Internal Bar Strength (IBS = (close-low)/(high-low)) with extreme readings (<0.15 for long, >0.85 for short) filtered by 12h EMA50 trend direction. Works in both bull and bear markets by only taking mean-reversion trades in the direction of the 12h trend, reducing false signals during strong moves. Targets 12-25 trades/year on 6h timeframe with discrete position sizing (0.0, ±0.25) to minimize fee drag.
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
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate IBS for mean reversion signals
    # IBS = (close - low) / (high - low), ranges 0 to 1
    # Avoid division by zero
    hl_range = high - low
    hl_range_safe = np.where(hl_range == 0, 1e-10, hl_range)
    ibs = (close - low) / hl_range_safe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of EMA50 (50)
    start_idx = 50
    
    for i in range(start_idx, n):
        ibs_val = ibs[i]
        trend_val = ema50_12h_aligned[i]
        close_val = close[i]
        
        # Skip if trend data not ready
        if np.isnan(trend_val):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price > 12h EMA50 = uptrend, price < 12h EMA50 = downtrend
        is_uptrend = close_val > trend_val
        is_downtrend = close_val < trend_val
        
        # IBS mean reversion conditions (extreme readings)
        long_signal = ibs_val < 0.15   # Oversold: close near low
        short_signal = ibs_val > 0.85  # Overbought: close near high
        
        # Entry conditions: IBS extreme in direction of 12h trend
        # In uptrend: look for oversold (long) opportunities
        # In downtrend: look for overbought (short) opportunities
        long_entry = long_signal and is_uptrend
        short_entry = short_signal and is_downtrend
        
        # Exit conditions: reverse IBS signal or trend change
        long_exit = ibs_val > 0.5 or not is_uptrend  # Exit when mean reverts or trend changes
        short_exit = ibs_val < 0.5 or not is_downtrend
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_IBS_Reversion_12hTrend_Filter"
timeframe = "6h"
leverage = 1.0