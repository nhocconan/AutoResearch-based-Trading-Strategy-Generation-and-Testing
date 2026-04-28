#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: On 4h timeframe, enter long when price breaks above R3 with volume surge and daily uptrend (EMA34 > EMA89), short when price breaks below S3 with volume surge and daily downtrend. Exit on opposite S1/R1 level. Uses Camarilla's most significant levels for institutional breakouts. Daily trend filter avoids counter-trend trades. Volume surge confirms institutional participation. Targets 20-40 trades/year to minimize fee drag.
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
    
    # Get daily data for trend filter and Camarilla calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 89:
        return np.zeros(n)
    
    # Calculate daily 34 and 89 EMA for trend filter
    close_daily = df_daily['close'].values
    ema34_daily = pd.Series(close_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_daily = pd.Series(close_daily).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align daily EMAs to 4h timeframe
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    ema89_daily_aligned = align_htf_to_ltf(prices, df_daily, ema89_daily)
    
    # Daily trend: bullish when EMA34 > EMA89
    daily_uptrend = ema34_daily_aligned > ema89_daily_aligned
    daily_downtrend = ema34_daily_aligned < ema89_daily_aligned
    
    # Previous day's data for Camarilla calculation
    prev_close = np.roll(close_daily, 1)
    prev_high = np.roll(df_daily['high'].values, 1)
    prev_low = np.roll(df_daily['low'].values, 1)
    prev_close[0] = close_daily[0]
    prev_high[0] = df_daily['high'].values[0]
    prev_low[0] = df_daily['low'].values[0]
    
    # Calculate Camarilla pivot levels
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    camarilla_range = prev_high - prev_low
    r3 = prev_close + camarilla_range * 1.1 / 4
    s3 = prev_close - camarilla_range * 1.1 / 4
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3)
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1)
    
    # Volume confirmation: current volume > 2.0x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_surge = volume > (vol_ma_50 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_daily_aligned[i]) or np.isnan(ema89_daily_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with daily trend alignment and volume surge
        long_entry = close[i] > r3_aligned[i] and daily_uptrend[i] and volume_surge[i]
        short_entry = close[i] < s3_aligned[i] and daily_downtrend[i] and volume_surge[i]
        
        # Exit on opposite Camarilla R1/S1 level (more conservative exit)
        long_exit = close[i] < s1_aligned[i]
        short_exit = close[i] > r1_aligned[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0