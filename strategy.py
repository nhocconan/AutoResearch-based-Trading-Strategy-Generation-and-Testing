#!/usr/bin/env python3
"""
4h_1d_Camarilla_Trend_Filter_v1
Hypothesis: Use daily trend from 1D close above/below EMA50 to filter 4H breakouts at Camarilla R4/S4 levels.
Go long when 4H close breaks above daily R4 AND daily trend is up (close > EMA50).
Go short when 4H close breaks below daily S4 AND daily trend is down (close < EMA50).
Exit when price returns to daily midpoint (R4+S4)/2.
Targets 20-40 trades per year to minimize fee drag. Works in bull/bear via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Trend_Filter_v1"
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
    
    # Daily data for trend and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    daily_close = df_1d['close'].values
    ema50 = np.full(len(daily_close), np.nan)
    if len(daily_close) >= 50:
        alpha = 2 / (50 + 1)
        ema50[0] = daily_close[0]
        for i in range(1, len(daily_close)):
            ema50[i] = alpha * daily_close[i] + (1 - alpha) * ema50[i-1]
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    range_ = prev_high - prev_low
    camarilla_r4 = prev_close + range_ * 1.1 / 2
    camarilla_s4 = prev_close - range_ * 1.1 / 2
    
    # Handle invalid ranges
    valid_range = range_ > 0
    camarilla_r4 = np.where(valid_range, camarilla_r4, np.nan)
    camarilla_s4 = np.where(valid_range, camarilla_s4, np.nan)
    
    # Align to 4h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(ema50_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: daily close above/below EMA50
        daily_close_price = df_1d['close'].values
        daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close_price)
        trend_up = daily_close_aligned[i] > ema50_aligned[i]
        
        # Breakout conditions
        long_breakout = close[i] > camarilla_r4_aligned[i] and trend_up
        short_breakout = close[i] < camarilla_s4_aligned[i] and not trend_up
        
        # Exit conditions: return to Camarilla midpoint
        camarilla_midpoint = (camarilla_r4_aligned[i] + camarilla_s4_aligned[i]) / 2
        
        long_exit = close[i] < camarilla_midpoint
        short_exit = close[i] > camarilla_midpoint
        
        # Signal logic
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals