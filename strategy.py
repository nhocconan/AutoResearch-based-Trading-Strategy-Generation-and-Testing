#!/usr/bin/env python3
# 1D_Camarilla_R3_S3_WeeklyTrend_VolumeBreakout
# Hypothesis: Daily chart strategy using weekly Camarilla R3/S3 breakouts with weekly trend filter (price > weekly EMA34) and volume spike confirmation.
# Uses weekly timeframe for trend and key levels to reduce noise and false breaks, while daily chart provides timely entries.
# Designed for low trade frequency (target 10-25 trades/year) to minimize fee drag and work in both bull and bear markets.
# Weekly trend filter ensures we only trade in direction of higher timeframe momentum.
# Volume spike confirms institutional participation at key levels.

name = "1D_Camarilla_R3_S3_WeeklyTrend_VolumeBreakout"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need enough data for weekly EMA34
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (based on previous week's OHLC)
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    # Calculate weekly Camarilla levels
    range_1w = prev_high - prev_low
    r3 = prev_close + range_1w * 1.1 / 4
    s3 = prev_close - range_1w * 1.1 / 4
    
    # Calculate weekly EMA34 for trend filter
    ema_34 = pd.Series(prev_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly Camarilla levels and EMA to daily timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Volume filter: current volume > 2.0x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure we have weekly EMA34 and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation (2.0x average volume)
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above weekly R3 + weekly uptrend (price > weekly EMA34) + volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_aligned[i] and   # Weekly uptrend filter
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S3 + weekly downtrend (price < weekly EMA34) + volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and   # Weekly downtrend filter
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit conditions:
            # 1. Price returns to weekly EMA34 (trend reversal signal)
            # 2. Opposite S3/R3 level break (tighter exit for profit protection)
            trend_reversal = (position == 1 and close[i] < ema_34_aligned[i]) or \
                           (position == -1 and close[i] > ema_34_aligned[i])
            opposite_break = (position == 1 and close[i] < s3_aligned[i]) or \
                           (position == -1 and close[i] > r3_aligned[i])
            
            if trend_reversal or opposite_break:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals