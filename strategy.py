#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1-day/1-week Camarilla pivot levels + volume confirmation.
# Uses weekly Camarilla levels for trend direction and daily levels for entry timing.
# Fades at daily S3/R3 in the direction of weekly trend and breaks out at daily S4/R4.
# Volume filter confirms institutional participation. Designed for 20-40 trades/year.
# Weekly trend filter reduces whipsaw in sideways markets and improves win rate.

name = "4h_1w1d_camarilla_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels for trend direction
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    daily_range_1w = prev_high_1w - prev_low_1w
    r4_1w = prev_close_1w + daily_range_1w * 1.1 / 2
    r3_1w = prev_close_1w + daily_range_1w * 1.1 / 4
    s3_1w = prev_close_1w - daily_range_1w * 1.1 / 4
    s4_1w = prev_close_1w - daily_range_1w * 1.1 / 2
    
    # Weekly trend: price above R3 = bullish, below S3 = bearish
    weekly_trend_bull = prev_close_1w > (r3_1w + s3_1w) / 2  # Above midpoint
    weekly_trend_bear = prev_close_1w < (r3_1w + s3_1w) / 2  # Below midpoint
    
    # Align weekly trend to 4h
    weekly_trend_bull_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_bull)
    weekly_trend_bear_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_bear)
    
    # Calculate daily Camarilla levels for entry
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    daily_range_1d = prev_high_1d - prev_low_1d
    r4_1d = prev_close_1d + daily_range_1d * 1.1 / 2
    r3_1d = prev_close_1d + daily_range_1d * 1.1 / 4
    s3_1d = prev_close_1d - daily_range_1d * 1.1 / 4
    s4_1d = prev_close_1d - daily_range_1d * 1.1 / 2
    
    # Daily average volume (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20 = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_avg_20[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align daily levels and volume to 4h
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(vol_avg_aligned[i]) or
            np.isnan(weekly_trend_bull_aligned[i]) or np.isnan(weekly_trend_bear_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * daily average volume
        vol_filter = volume[i] > 1.5 * vol_avg_aligned[i]
        
        # Determine weekly trend direction
        is_bullish_week = weekly_trend_bull_aligned[i]
        is_bearish_week = weekly_trend_bear_aligned[i]
        
        # Fade at S3/R3 in direction of weekly trend
        fade_long = (low[i] <= s3_1d_aligned[i] and vol_filter and is_bullish_week)
        fade_short = (high[i] >= r3_1d_aligned[i] and vol_filter and is_bearish_week)
        
        # Breakout at S4/R4 (always active, but stronger with trend)
        breakout_long = (high[i] >= r4_1d_aligned[i] and vol_filter)
        breakout_short = (low[i] <= s4_1d_aligned[i] and vol_filter)
        
        # Exit when price returns to weekly midpoint or opposite S3/R3
        weekly_midpoint = (r3_1w + s3_1w) / 2
        weekly_midpoint_aligned = align_htf_to_ltf(prices, df_1w, weekly_midpoint)
        
        exit_long = (position == 1 and 
                    (low[i] <= weekly_midpoint_aligned[i] or 
                     high[i] >= s3_1d_aligned[i]))  # Exit long if hits weekly midpoint or daily S3
        exit_short = (position == -1 and 
                     (high[i] >= weekly_midpoint_aligned[i] or 
                      low[i] <= r3_1d_aligned[i]))  # Exit short if hits weekly midpoint or daily R3
        
        # Priority: breakout > fade > hold
        if breakout_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif fade_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif fade_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals