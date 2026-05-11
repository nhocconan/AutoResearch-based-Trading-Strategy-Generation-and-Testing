#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeS
Hypothesis: Trade breakouts at weekly Camarilla R1/S1 levels with 1-week trend filter and volume confirmation on 12h timeframe.
Uses weekly higher timeframe for trend direction (avoids whipsaw in sideways markets) and daily for pivot calculation to reduce noise.
Target: 15-30 trades/year on 12h (~60-120 total over 4 years) to minimize fee drag while capturing significant breaks.
Works in bull/bear by aligning with weekly trend - only takes longs in weekly uptrend, shorts in weekly downtrend.
"""

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeS"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly OHLC for Camarilla Pivots (using 1d data to calculate weekly levels) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Resample 1d to weekly manually for pivot calculation (using actual weekly aggregation logic)
    # We'll calculate weekly OHLC from daily data: weekly high = max of daily highs, etc.
    weekly_high = []
    weekly_low = []
    weekly_close = []
    
    # Simple approach: use every 5th day as weekly approximation (5 trading days ≈ 1 week)
    # More robust: group by actual week number
    for i in range(0, len(df_1d), 5):
        if i + 5 <= len(df_1d):
            week_high = np.max(df_1d['high'].iloc[i:i+5])
            week_low = np.min(df_1d['low'].iloc[i:i+5])
            week_close = df_1d['close'].iloc[i+4]  # last day of the week
            weekly_high.append(week_high)
            weekly_low.append(week_low)
            weekly_close.append(week_close)
    
    if len(weekly_high) < 2:
        return np.zeros(n)
    
    weekly_high = np.array(weekly_high)
    weekly_low = np.array(weekly_low)
    weekly_close = np.array(weekly_close)
    
    # Calculate Camarilla levels from previous week's OHLC
    wkly_ph = weekly_high[:-1]  # previous week high
    wkly_pl = weekly_low[:-1]   # previous week low
    wkly_pc = weekly_close[:-1] # previous week close
    
    # Camarilla R1/S1 from previous week
    camarilla_r1_weekly = wkly_pc + (wkly_ph - wkly_pl) * 1.1 / 2
    camarilla_s1_weekly = wkly_pc - (wkly_ph - wkly_pl) * 1.1 / 2
    
    # Need to align weekly data to 1d first, then to 12h
    # Create arrays matching df_1d length for alignment
    wkly_r1_aligned_1d = np.full(len(df_1d), np.nan)
    wkly_s1_aligned_1d = np.full(len(df_1d), np.nan)
    
    # Map weekly values to daily indices (each weekly value applies to 5 days)
    for i in range(len(wkly_pc)):
        start_idx = i * 5
        end_idx = min(start_idx + 5, len(df_1d))
        if start_idx < len(df_1d):
            wkly_r1_aligned_1d[start_idx:end_idx] = camarilla_r1_weekly[i]
            wkly_s1_aligned_1d[start_idx:end_idx] = camarilla_s1_weekly[i]
    
    # Now align from 1d to 12h
    r1_12h = align_htf_to_ltf(prices, df_1d, wkly_r1_aligned_1d)
    s1_12h = align_htf_to_ltf(prices, df_1d, wkly_s1_aligned_1d)
    
    # === Weekly Trend Filter (EMA34 on weekly close) ===
    wkly_ema34 = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align weekly EMA to daily then to 12h
    wkly_ema34_aligned_1d = np.full(len(df_1d), np.nan)
    for i in range(len(wkly_ema34)):
        start_idx = i * 5
        end_idx = min(start_idx + 5, len(df_1d))
        if start_idx < len(df_1d):
            wkly_ema34_aligned_1d[start_idx:end_idx] = wkly_ema34[i]
    
    ema34_12h = align_htf_to_ltf(prices, df_1d, wkly_ema34_aligned_1d)
    
    # === Volume Filter (1.5x 20-period EMA on 12h) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or np.isnan(ema34_12h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above weekly R1 with weekly uptrend and volume
            if (close[i] > r1_12h[i] and 
                close[i] > ema34_12h[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below weekly S1 with weekly downtrend and volume
            elif (close[i] < s1_12h[i] and 
                  close[i] < ema34_12h[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below weekly S1 (reversal) or weekly trend turns down
            if close[i] < s1_12h[i] or close[i] < ema34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price breaks above weekly R1 (reversal) or weekly trend turns up
            if close[i] > r1_12h[i] or close[i] > ema34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals