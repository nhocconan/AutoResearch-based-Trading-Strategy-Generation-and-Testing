#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_WeeklyTrend_Volume
Hypothesis: On 12-hour timeframe, use Camarilla R3/S3 breakouts in the direction of weekly trend (via weekly EMA8/EMA21) with volume confirmation. Weekly trend filter avoids counter-trend trades during extended trends, while Camarilla breakouts capture momentum bursts at key intraday levels. Volume surge confirms institutional participation. Designed for low trade frequency (~12-30/year) to minimize fee decay and improve generalization across bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 21:
        return np.zeros(n)
    
    # Calculate weekly 8 and 21 EMA for trend filter
    close_weekly = df_weekly['close'].values
    ema8_weekly = pd.Series(close_weekly).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21_weekly = pd.Series(close_weekly).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align weekly EMAs to 12h timeframe
    ema8_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema8_weekly)
    ema21_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema21_weekly)
    
    # Weekly trend: bullish when EMA8 > EMA21
    weekly_uptrend = ema8_weekly_aligned > ema21_weekly_aligned
    weekly_downtrend = ema8_weekly_aligned < ema21_weekly_aligned
    
    # Get daily data for Camarilla levels
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_daily['high'].shift(1).values
    low_prev = df_daily['low'].shift(1).values
    close_prev = df_daily['close'].shift(1).values
    
    # Camarilla levels: R3, S3
    R3 = close_prev + (high_prev - low_prev) * 1.1 / 4
    S3 = close_prev - (high_prev - low_prev) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_daily, R3)
    S3_aligned = align_htf_to_ltf(prices, df_daily, S3)
    
    # Camarilla breakouts
    breakout_long = close > R3_aligned
    breakout_short = close < S3_aligned
    
    # Volume confirmation: current volume > 1.5x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_surge = volume > (vol_ma_30 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema8_weekly_aligned[i]) or np.isnan(ema21_weekly_aligned[i]) or
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with weekly trend alignment and volume surge
        long_entry = breakout_long[i] and weekly_uptrend[i] and volume_surge[i]
        short_entry = breakout_short[i] and weekly_downtrend[i] and volume_surge[i]
        
        # Exit on opposite breakout with volume surge
        long_exit = breakout_short[i] and volume_surge[i]
        short_exit = breakout_long[i] and volume_surge[i]
        
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

name = "12h_Camarilla_R3_S3_Breakout_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0