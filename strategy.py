#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_Volume_EMAFilter_v1
Hypothesis: Trade Camarilla R1/S1 breakouts on daily timeframe with EMA50 trend filter and volume confirmation.
In bull markets: buy R1 breakouts above EMA50. In bear markets: short S3 breakdowns below EMA50.
Uses weekly pivot for long-term bias. Designed for low trade frequency (<25/year) to minimize fee drag.
Works in both bull/bear via adaptive entry rules based on higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R1, S1, R3, S3
    rang = prev_high - prev_low
    r1 = prev_close + rang * 1.0 / 12
    s1 = prev_close - rang * 1.0 / 12
    r3 = prev_close + rang * 3.0 / 12
    s3 = prev_close - rang * 3.0 / 12
    
    # Align to 1d timeframe (no additional delay needed for same timeframe)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Load 1w data for weekly pivot bias (long-term direction)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point: (H+L+C)/3
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    weekly_pivot = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(weekly_pivot_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.3 * vol_ma
        else:
            volume_ok = False
        
        # Determine weekly bias: price above weekly pivot = bullish
        weekly_bullish = price > weekly_pivot_aligned[i]
        
        if position == 0:
            # Long conditions: price > R1 AND price > EMA50 AND weekly bullish AND volume
            if (price > r1_aligned[i] and 
                price > ema_50_1d_aligned[i] and 
                weekly_bullish and 
                volume_ok):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < S1 AND price < EMA50 AND weekly bearish AND volume
            elif (price < s1_aligned[i] and 
                  price < ema_50_1d_aligned[i] and 
                  not weekly_bullish and 
                  volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < EMA50 or price < S1 (mean reversion) or weekly bias turns bearish
            if (price < ema_50_1d_aligned[i] or 
                price < s1_aligned[i] or 
                not weekly_bullish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > EMA50 or price > R1 (mean reversion) or weekly bias turns bullish
            if (price > ema_50_1d_aligned[i] or 
                price > r1_aligned[i] or 
                weekly_bullish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_Volume_EMAFilter_v1"
timeframe = "1d"
leverage = 1.0