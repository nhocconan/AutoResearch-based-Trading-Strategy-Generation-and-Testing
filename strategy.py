#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_WeeklyTrend_VolumeSpike
Hypothesis: On 12-hour timeframe, use daily Camarilla pivot levels (R1/S1) as support/resistance.
Enter long when price breaks above R1 with volume surge and weekly uptrend (EMA8 > EMA21),
short when price breaks below S1 with volume surge and weekly downtrend.
Exit on opposite Camarilla level break with volume surge.
Designed for low trade frequency (~12-30/year) to minimize fee decay in both bull and bear markets.
Weekly trend filter avoids counter-trend trades during extended trends, while Camarilla levels
provide institutional reference points. Volume surge confirms institutional participation.
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
    
    # Get daily data for Camarilla calculation (using previous day's OHLC)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close_daily = df_daily['close'].values
    prev_high_daily = df_daily['high'].values
    prev_low_daily = df_daily['low'].values
    
    # Align daily data to 12h timeframe
    prev_close_aligned = align_htf_to_ltf(prices, df_daily, prev_close_daily)
    prev_high_aligned = align_htf_to_ltf(prices, df_daily, prev_high_daily)
    prev_low_aligned = align_htf_to_ltf(prices, df_daily, prev_low_daily)
    
    # Calculate Camarilla pivot levels
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    camarilla_range = prev_high_aligned - prev_low_aligned
    r1 = prev_close_aligned + camarilla_range * 1.1 / 12
    s1 = prev_close_aligned - camarilla_range * 1.1 / 12
    
    # Volume confirmation: current volume > 2.0x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_surge = volume > (vol_ma_50 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema8_weekly_aligned[i]) or np.isnan(ema21_weekly_aligned[i]) or
            np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with weekly trend alignment and volume surge
        long_entry = close[i] > r1[i] and weekly_uptrend[i] and volume_surge[i]
        short_entry = close[i] < s1[i] and weekly_downtrend[i] and volume_surge[i]
        
        # Exit on opposite Camarilla level break with volume surge
        long_exit = close[i] < s1[i] and volume_surge[i]
        short_exit = close[i] > r1[i] and volume_surge[i]
        
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

name = "12h_Camarilla_R1_S1_WeeklyTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0