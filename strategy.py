#!/usr/bin/env python3
"""
6h 20-bar Donchian breakout with weekly pivot direction and volume confirmation
Hypothesis: Donchian breakouts capture institutional momentum, filtered by weekly pivot
direction for long-term bias and volume confirmation for conviction. Works in bull
(buy breakouts above weekly pivot) and bear (sell breakdowns below weekly pivot).
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR
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
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    weekly_pivot = np.full(len(weekly_close), np.nan)
    weekly_r1 = np.full(len(weekly_close), np.nan)
    weekly_s1 = np.full(len(weekly_close), np.nan)
    weekly_r2 = np.full(len(weekly_close), np.nan)
    weekly_s2 = np.full(len(weekly_close), np.nan)
    weekly_r3 = np.full(len(weekly_close), np.nan)
    weekly_s3 = np.full(len(weekly_close), np.nan)
    
    for i in range(len(weekly_close)):
        if not (np.isnan(weekly_high[i]) or np.isnan(weekly_low[i]) or np.isnan(weekly_close[i])):
            pivot = (weekly_high[i] + weekly_low[i] + weekly_close[i]) / 3.0
            weekly_pivot[i] = pivot
            weekly_r1[i] = 2 * pivot - weekly_low[i]
            weekly_s1[i] = 2 * pivot - weekly_high[i]
            weekly_r2[i] = pivot + (weekly_high[i] - weekly_low[i])
            weekly_s2[i] = pivot - (weekly_high[i] - weekly_low[i])
            weekly_r3[i] = weekly_high[i] + 2 * (pivot - weekly_low[i])
            weekly_s3[i] = weekly_low[i] - 2 * (weekly_high[i] - pivot)
    
    # Get daily data for volume confirmation
    df_daily = get_htf_data(prices, '1d')
    daily_volume = df_daily['volume'].values
    
    # 20-period average volume on daily
    vol_ma_daily = np.full(len(daily_volume), np.nan)
    for i in range(20, len(daily_volume)):
        vol_ma_daily[i] = np.mean(daily_volume[i-20:i])
    
    # Align indicators to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s3)
    vol_ma_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_daily)
    
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
        if (np.isnan(atr[i]) or np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(weekly_r3_aligned[i]) or np.isnan(weekly_s3_aligned[i]) or
            np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(vol_ma_daily_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 6h volume > 1.5x daily average volume (scaled)
        # Scale daily volume to 6h: approx 1/4 of daily volume (since 4x 6h in 1d)
        vol_threshold = vol_ma_daily_aligned[i] / 4.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR below weekly S3
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lower[i] or
                close[i] < weekly_s3_aligned[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR above weekly R3
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > upper[i] or
                close[i] > weekly_r3_aligned[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 12 bars flat
            if bars_since_entry >= 12:
                # Breakout entries: upper/lower with weekly pivot filter
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                
                # Long: breakout above upper with price above weekly pivot + volume
                if bull_breakout and close[i] > weekly_pivot_aligned[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below lower with price below weekly pivot + volume
                elif bear_breakout and close[i] < weekly_pivot_aligned[i] and volume_filter:
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