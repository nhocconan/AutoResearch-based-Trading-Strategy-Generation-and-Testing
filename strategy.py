#!/usr/bin/env python3
"""
6h_1d_Weekly_Pivot_Directional
Hypothesis: Trade in the direction of weekly pivot trend using 1d open position relative to weekly pivot.
In bull markets, price tends to open above weekly pivot and stay bullish; in bear markets, opens below and stays bearish.
Uses 1d open vs weekly pivot (PP) as trend filter, with 6h EMA(20) for entry timing.
Works in both bull (buy dips above PP) and bear (sell rallies below PP) markets. Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot points (standard floor method)."""
    PP = (high + low + close) / 3.0
    R1 = 2 * PP - low
    S1 = 2 * PP - high
    R2 = PP + (high - low)
    S2 = PP - (high - low)
    R3 = high + 2 * (PP - low)
    S3 = low - 2 * (high - PP)
    return PP, R1, R2, R3, S1, S2, S3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    PP_weekly, R1_weekly, R2_weekly, R3_weekly, S1_weekly, S2_weekly, S3_weekly = calculate_weekly_pivot(
        weekly_high, weekly_low, weekly_close
    )
    
    # Align weekly pivot to 6h (only PP needed for trend filter)
    PP_weekly_aligned = align_htf_to_ltf(prices, df_weekly, PP_weekly)
    
    # 6h EMA for entry timing
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if PP data not ready
        if np.isnan(PP_weekly_aligned[i]) or np.isnan(ema_20[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d open vs weekly PP
        # Get the 1d open price for current day (using same index mapping as weekly)
        # Since we're on 6h chart, we need to check if current bar is the first 6h bar of the day
        # Simpler approach: use the relationship between current open and weekly PP
        # If weekly PP is trending up (current open > PP), look for longs on dips to EMA
        # If weekly PP is trending down (current open < PP), look for shorts on rallies to EMA
        
        # Determine weekly trend based on price position relative to PP
        if open_price[i] > PP_weekly_aligned[i]:
            # Bullish bias: look for longs when price dips to EMA support
            if close[i] <= ema_20[i] and close[i] > PP_weekly_aligned[i] * 0.995:  # near EMA but above PP
                if position != 1:
                    position = 1
                    signals[i] = position_size
            # Exit long if price breaks below PP
            elif position == 1 and close[i] < PP_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size if position == 1 else 0.0
        else:
            # Bearish bias: look for shorts when price rallies to EMA resistance
            if close[i] >= ema_20[i] and close[i] < PP_weekly_aligned[i] * 1.005:  # near EMA but below PP
                if position != -1:
                    position = -1
                    signals[i] = -position_size
            # Exit short if price breaks above PP
            elif position == -1 and close[i] > PP_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size if position == -1 else 0.0
    
    return signals

name = "6h_1d_Weekly_Pivot_Directional"
timeframe = "6h"
leverage = 1.0