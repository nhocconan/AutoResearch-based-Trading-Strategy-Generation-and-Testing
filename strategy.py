#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_Volume
Hypothesis: On 12-hour timeframe, enter long when price breaks above Camarilla R3 level with volume surge and weekly uptrend (price above weekly EMA34), short when price breaks below S3 level with volume surge and weekly downtrend. Exit on opposite breakout. Uses weekly trend filter to avoid counter-trend trades. Designed for low trade frequency (~15-30/year) to minimize fee decay in both bull and bear markets.
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 34:
        return np.zeros(n)
    
    # Calculate weekly 34 EMA for trend filter
    close_weekly = df_weekly['close'].values
    ema34_weekly = pd.Series(close_weekly).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA to 12h timeframe
    ema34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Weekly trend: bullish when price > EMA34, bearish when price < EMA34
    weekly_uptrend = close > ema34_weekly_aligned
    weekly_downtrend = close < ema34_weekly_aligned
    
    # Get daily data for Camarilla levels
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_daily['high'].shift(1).values
    low_prev = df_daily['low'].shift(1).values
    close_prev = df_daily['close'].shift(1).values
    
    # Camarilla formulas
    range_prev = high_prev - low_prev
    R3 = close_prev + range_prev * 1.1 / 2
    S3 = close_prev - range_prev * 1.1 / 2
    
    # Align daily Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_daily, R3)
    S3_aligned = align_htf_to_ltf(prices, df_daily, S3)
    
    # Volume confirmation: current volume > 2.0x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_surge = volume > (vol_ma_24 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_weekly_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with weekly trend alignment and volume surge
        long_entry = close[i] > R3_aligned[i] and weekly_uptrend[i] and volume_surge[i]
        short_entry = close[i] < S3_aligned[i] and weekly_downtrend[i] and volume_surge[i]
        
        # Exit on opposite Camarilla level break with volume surge
        long_exit = close[i] < S3_aligned[i] and volume_surge[i]
        short_exit = close[i] > R3_aligned[i] and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0