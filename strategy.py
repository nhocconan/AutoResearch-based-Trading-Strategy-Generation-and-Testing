#!/usr/bin/env python3
"""
1h_4h_Camarilla_Breakout_With_Trend_Filter_v1
Hypothesis: Trade 1-hour bars using 4-hour Camarilla H3/L3 breakouts with trend filter.
Use 4h for signal direction (trend + Camarilla levels), 1h only for precise entry timing.
Add session filter (08-20 UTC) to reduce noise trades. Target 15-37 trades/year per symbol.
Works in bull markets (trend continuation breaks) and bear markets (mean reversion from extremes).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_Camarilla_Breakout_With_Trend_Filter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4H DATA FOR CAMARILLA PIVOTS ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels from previous 4h bar
    typical_price = (high_4h + low_4h + close_4h) / 3
    pivot = typical_price
    range_4h = high_4h - low_4h
    
    h3 = pivot + (range_4h * 1.1 / 4)
    l3 = pivot - (range_4h * 1.1 / 4)
    h4 = pivot + (range_4h * 1.1 / 2)
    l4 = pivot - (range_4h * 1.1 / 2)
    
    # Align to 1h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3)
    h4_aligned = align_htf_to_ltf(prices, df_4h, h4)
    l4_aligned = align_htf_to_ltf(prices, df_4h, l4)
    
    # === TREND FILTER: 21 EMA ON 4H CLOSE ===
    close_4h_series = pd.Series(close_4h)
    ema21_4h = close_4h_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # === SESSION FILTER: 08-20 UTC ===
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(ema21_4h_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Determine 4h trend
        uptrend = close_4h_aligned[i] > ema21_4h_aligned[i]  # Use 4h close aligned
        downtrend = close_4h_aligned[i] < ema21_4h_aligned[i]
        
        # Long: price breaks H3 with 4h uptrend
        long_signal = (close[i] > h3_aligned[i] and uptrend)
        
        # Short: price breaks L3 with 4h downtrend
        short_signal = (close[i] < l3_aligned[i] and downtrend)
        
        # Exit: opposite H3/L3 level or trend reversal
        exit_long = (position == 1 and 
                    (close[i] < l3_aligned[i] or not uptrend))
        exit_short = (position == -1 and 
                     (close[i] > h3_aligned[i] or not downtrend))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.20
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals