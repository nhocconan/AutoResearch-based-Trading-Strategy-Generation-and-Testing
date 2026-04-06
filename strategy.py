#!/usr/bin/env python3
"""
6h Weekly Pivot + Donchian(10) Breakout with Volume Confirmation
Hypothesis: Weekly pivots establish institutional support/resistance zones. 
Donchian breakouts capture momentum when price breaches these zones with volume confirmation.
Works in bull markets (breakouts above weekly resistance) and bear markets (breakdowns below weekly support).
Target: 80-150 total trades over 4 years (20-38/year) to balance opportunity and cost.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weeklypivot_donchian10_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Weekly data for pivot points and Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points (standard formula)
    # PP = (H + L + C) / 3
    # R1 = 2*PP - L
    # S1 = 2*PP - H
    # R2 = PP + (H - L)
    # S2 = PP - (H - L)
    # R3 = H + 2*(PP - L)
    # S3 = L - 2*(H - PP)
    weekly_pivot = np.full_like(weekly_close, np.nan)
    weekly_r1 = np.full_like(weekly_close, np.nan)
    weekly_s1 = np.full_like(weekly_close, np.nan)
    weekly_r2 = np.full_like(weekly_close, np.nan)
    weekly_s2 = np.full_like(weekly_close, np.nan)
    weekly_r3 = np.full_like(weekly_close, np.nan)
    weekly_s3 = np.full_like(weekly_close, np.nan)
    
    for i in range(len(weekly_close)):
        if not (np.isnan(weekly_high[i]) or np.isnan(weekly_low[i]) or np.isnan(weekly_close[i])):
            weekly_pivot[i] = (weekly_high[i] + weekly_low[i] + weekly_close[i]) / 3.0
            weekly_r1[i] = 2 * weekly_pivot[i] - weekly_low[i]
            weekly_s1[i] = 2 * weekly_pivot[i] - weekly_high[i]
            weekly_r2[i] = weekly_pivot[i] + (weekly_high[i] - weekly_low[i])
            weekly_s2[i] = weekly_pivot[i] - (weekly_high[i] - weekly_low[i])
            weekly_r3[i] = weekly_high[i] + 2 * (weekly_pivot[i] - weekly_low[i])
            weekly_s3[i] = weekly_low[i] - 2 * (weekly_high[i] - weekly_pivot[i])
    
    # Weekly Donchian channels (10-period) for breakout confirmation
    weekly_high_10 = np.full_like(weekly_high, np.nan)
    weekly_low_10 = np.full_like(weekly_low, np.nan)
    
    for i in range(10, len(weekly_high)):
        weekly_high_10[i] = np.max(weekly_high[i-10:i])
        weekly_low_10[i] = np.min(weekly_low[i-10:i])
    
    # Align weekly data to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s2)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s3)
    high_10_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high_10)
    low_10_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low_10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20  # Need enough data for calculations
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(high_10_aligned[i]) or 
            np.isnan(low_10_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[max(0, i-20):i])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below S1 OR against weekly pivot
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < s1_aligned[i] or
                close[i] < pivot_aligned[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks above R1 OR against weekly pivot
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > r1_aligned[i] or
                close[i] > pivot_aligned[i] or
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
                # Breakout entries: weekly pivot levels with Donchian confirmation
                # Long: price breaks above R1 with weekly Donchian confirmation + volume
                bull_breakout = close[i] > r1_aligned[i] and close[i] > high_10_aligned[i]
                # Short: price breaks below S1 with weekly Donchian confirmation + volume
                bear_breakout = close[i] < s1_aligned[i] and close[i] < low_10_aligned[i]
                
                # Long: breakout above R1 with volume
                if bull_breakout and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below S1 with volume
                elif bear_breakout and volume_filter:
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