#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend
Hypothesis: 1h Camarilla breakouts filtered by 4h trend direction for improved accuracy.
Goes long when price breaks above R1 in uptrend (4h close > 4h open), short when breaks below S1 in downtrend.
Session filter (08-20 UTC) to avoid low-liquidity hours. Position size fixed at 0.20 to manage risk.
Designed for moderate trade frequency (~15-30 trades/year) to balance opportunity with fee drag.
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h trend: bullish if close > open, bearish if close < open
    open_4h = df_4h['open'].values
    close_4h = df_4h['close'].values
    trend_bullish = close_4h > open_4h
    trend_bearish = close_4h < open_4h
    
    # Align 4h trend to 1h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_4h, trend_bullish.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_4h, trend_bearish.astype(float))
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla R1 and S1 levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r1 = close_1d + 0.4 * (high_1d - low_1d)
    camarilla_s1 = close_1d - 0.4 * (high_1d - low_1d)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(trend_bullish_aligned[i]) or 
            np.isnan(trend_bearish_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_r1 = close[i] > camarilla_r1_aligned[i]
        breakdown_s1 = close[i] < camarilla_s1_aligned[i]
        
        # Entry logic: breakout in direction of 4h trend during session
        long_entry = trend_bullish_aligned[i] > 0.5 and breakout_r1
        short_entry = trend_bearish_aligned[i] > 0.5 and breakdown_s1
        
        # Exit logic: opposite breakout or trend reversal
        long_exit = breakdown_s1 or (trend_bearish_aligned[i] > 0.5)
        short_exit = breakout_r1 or (trend_bullish_aligned[i] > 0.5)
        
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
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
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend"
timeframe = "1h"
leverage = 1.0