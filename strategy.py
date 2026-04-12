#!/usr/bin/env python3
"""
12h_1w_Camarilla_Trend_Reversal_v1
Hypothesis: Use weekly trend from 1W close above/below EMA20 to filter 12H reversals at Camarilla R3/S3 levels.
Go long when 12H close crosses below weekly S3 AND weekly trend is up (close > EMA20) - buying weakness in uptrend.
Go short when 12H close crosses above weekly R3 AND weekly trend is down (close < EMA20) - selling strength in downtrend.
Exit when price returns to weekly midpoint (R3+S3)/2.
Targets 15-25 trades per year to minimize fee drift. Works in bull/bear via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_Camarilla_Trend_Reversal_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly data for trend and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    weekly_close = df_1w['close'].values
    ema20 = np.full(len(weekly_close), np.nan)
    if len(weekly_close) >= 20:
        alpha = 2 / (20 + 1)
        ema20[0] = weekly_close[0]
        for i in range(1, len(weekly_close)):
            ema20[i] = alpha * weekly_close[i] + (1 - alpha) * ema20[i-1]
    
    # Previous week's OHLC for Camarilla calculation
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    # Calculate Camarilla levels
    range_ = prev_high - prev_low
    camarilla_r3 = prev_close + range_ * 1.1 / 4
    camarilla_s3 = prev_close - range_ * 1.1 / 4
    
    # Handle invalid ranges
    valid_range = range_ > 0
    camarilla_r3 = np.where(valid_range, camarilla_r3, np.nan)
    camarilla_s3 = np.where(valid_range, camarilla_s3, np.nan)
    
    # Align to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    ema20_aligned = align_htf_to_ltf(prices, df_1w, ema20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema20_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: weekly close above/below EMA20
        weekly_close_price = df_1w['close'].values
        weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close_price)
        trend_up = weekly_close_aligned[i] > ema20_aligned[i]
        
        # Reversal conditions: buying weakness in uptrid, selling strength in downtrend
        long_reversal = close[i] < camarilla_s3_aligned[i] and trend_up
        short_reversal = close[i] > camarilla_r3_aligned[i] and not trend_up
        
        # Exit conditions: return to Camarilla midpoint
        camarilla_midpoint = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2
        
        long_exit = close[i] > camarilla_midpoint
        short_exit = close[i] < camarilla_midpoint
        
        # Signal logic
        if long_reversal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_reversal and position != -1:
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