#!/usr/bin/env python3
"""
6h Weekly Pivot + Donchian Breakout with Volume Confirmation
Hypothesis: Weekly pivot points provide strong support/resistance levels. 
When price breaks Donchian(20) channels near weekly pivot levels with volume confirmation,
it indicates institutional interest and trend continuation. Works in both bull and bear markets
by using weekly pivot direction as trend filter.
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_donchian20_vol"
timeframe = "6h"
leverage = 1.0

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot points from previous week's OHLC"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stoploss and filters
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    weekly_volume = df_weekly['volume'].values
    
    # Calculate weekly pivot points
    pivot_points = np.full(len(weekly_close), np.nan)
    r1_points = np.full(len(weekly_close), np.nan)
    s1_points = np.full(len(weekly_close), np.nan)
    r2_points = np.full(len(weekly_close), np.nan)
    s2_points = np.full(len(weekly_close), np.nan)
    r3_points = np.full(len(weekly_close), np.nan)
    s3_points = np.full(len(weekly_close), np.nan)
    
    for i in range(1, len(weekly_close)):
        pivot, r1, r2, r3, s1, s2, s3 = calculate_weekly_pivot(
            weekly_high[i-1], weekly_low[i-1], weekly_close[i-1]
        )
        pivot_points[i] = pivot
        r1_points[i] = r1
        s1_points[i] = s1
        r2_points[i] = r2
        s2_points[i] = s2
        r3_points[i] = r3
        s3_points[i] = s3
    
    # Align weekly pivot data to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot_points)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1_points)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1_points)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2_points)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2_points)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3_points)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3_points)
    
    # Weekly trend: price above/below pivot
    weekly_trend = np.where(weekly_close > pivot_points, 1, -1)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_weekly, weekly_trend)
    
    # Weekly volume average
    vol_ma_weekly = np.full(len(weekly_volume), np.nan)
    for i in range(4, len(weekly_volume)):  # 4-week average
        vol_ma_weekly[i] = np.mean(weekly_volume[i-4:i])
    vol_ma_weekly_aligned = align_htf_to_ltf(prices, df_weekly, vol_ma_weekly)
    
    # Donchian channels (20-period) from 6h data
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 40  # Need enough data for Donchian and alignments
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(weekly_trend_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(vol_ma_weekly_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 6h volume > 1.3x weekly average volume (scaled)
        # Scale weekly volume to 6h: approx 1/28 of weekly volume (28x 6h in 1 week)
        vol_threshold = vol_ma_weekly_aligned[i] / 28.0 * 1.3
        volume_filter = volume[i] > vol_threshold
        
        # Session filter: 08-20 UTC (most active trading hours)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        session_filter = 8 <= hour <= 20
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR against weekly trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lower[i] or
                weekly_trend_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR against weekly trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > upper[i] or
                weekly_trend_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 8 bars flat
            if bars_since_entry >= 8:
                # Breakout entries: upper/lower with weekly trend
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                
                # Additional filters: price near pivot levels (within 0.5% of R1/S1 or R2/S2)
                near_r1 = abs(close[i] - r1_aligned[i]) / close[i] < 0.005
                near_s1 = abs(close[i] - s1_aligned[i]) / close[i] < 0.005
                near_r2 = abs(close[i] - r2_aligned[i]) / close[i] < 0.005
                near_s2 = abs(close[i] - s2_aligned[i]) / close[i] < 0.005
                near_pivot_level = near_r1 or near_s1 or near_r2 or near_s2
                
                # Long: breakout above upper with bullish weekly trend + volume + session + near pivot
                if bull_breakout and weekly_trend_aligned[i] == 1 and volume_filter and session_filter and near_pivot_level:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below lower with bearish weekly trend + volume + session + near pivot
                elif bear_breakout and weekly_trend_aligned[i] == -1 and volume_filter and session_filter and near_pivot_level:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals