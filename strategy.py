#!/usr/bin/env python3
"""
4h_1d_Turtle_Soup_Reversal_v1
Hypothesis: Turtle Soup reversal pattern at 4h timeframe using 1d high/low liquidity sweeps.
Looks for false breakouts of prior day's high/low with rejection and confluence with 1d EMA trend.
Designed for low trade frequency (target 20-40/year) by requiring:
1. Liquidity sweep (false breakout of 1d high/low)
2. Price rejection (close inside prior day's range)
3. Trend alignment (price vs 1d EMA)
Works in bull/bear markets by fading false breakouts with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Turtle_Soup_Reversal_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE before loop for liquidity levels and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high and low (liquidity levels)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # 1-day range for context
    prev_range = prev_high - prev_low
    
    # 1d EMA (21 period) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 1d data to 4h timeframe
    prev_high_4h = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_4h = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_4h = align_htf_to_ltf(prices, df_1d, prev_close)
    prev_range_4h = align_htf_to_ltf(prices, df_1d, prev_range)
    ema_1d_4h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(prev_high_4h[i]) or np.isnan(prev_low_4h[i]) or 
            np.isnan(ema_1d_4h[i]) or np.isnan(prev_range_4h[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Liquidity sweep detection: false breakout of prior day's high/low
        # Long setup: price breaks below prior day's low but closes back inside range
        # Short setup: price breaks above prior day's high but closes back inside range
        
        # Check for false breakdown (long setup)
        false_breakdown = (low[i] < prev_low_4h[i]) and (close[i] > prev_low_4h[i])
        
        # Check for false breakout (short setup)
        false_breakout = (high[i] > prev_high_4h[i]) and (close[i] < prev_high_4h[i])
        
        # Rejection strength: how far price recovered back into the range
        breakdown_recovery = (close[i] - low[i]) / (prev_high_4h[i] - prev_low_4h[i] + 1e-10)
        breakout_rejection = (high[i] - close[i]) / (prev_high_4h[i] - prev_low_4h[i] + 1e-10)
        
        # Minimum rejection threshold (at least 30% recovery back into range)
        min_rejection = 0.3
        strong_breakdown_rejection = breakdown_recovery > min_rejection
        strong_breakout_rejection = breakout_rejection > min_rejection
        
        # Trend filter: align with 1d EMA direction
        # In uptrend (price > EMA), look for long setups on false breakdowns
        # In downtrend (price < EMA), look for short setups on false breakouts
        uptrend = close[i] > ema_1d_4h[i]
        downtrend = close[i] < ema_1d_4h[i]
        
        # Entry conditions
        long_entry = false_breakdown and strong_breakdown_rejection and uptrend
        short_entry = false_breakout and strong_breakout_rejection and downtrend
        
        # Exit conditions: return to prior day's close or trend reversal
        long_exit = (close[i] < prev_close_4h[i]) or (close[i] < ema_1d_4h[i])
        short_exit = (close[i] > prev_close_4h[i]) or (close[i] > ema_1d_4h[i])
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals