#!/usr/bin/env python3
"""
6h Weekly Pivot + Volume Confirmation Strategy
Hypothesis: Weekly pivot levels act as strong support/resistance. Price breaking above weekly R3 with volume confirmation signals institutional buying (works in bull), breaking below S3 signals distribution (works in bear). Uses 6h timeframe to reduce noise while capturing multi-day moves. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_volume_v1"
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
    
    # 14-period ATR for stoploss
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
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H + L + C)/3
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    pivot = (high_weekly + low_weekly + close_weekly) / 3.0
    
    # Weekly support/resistance levels
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    r1 = 2 * pivot - low_weekly
    s1 = 2 * pivot - high_weekly
    r2 = pivot + (high_weekly - low_weekly)
    s2 = pivot - (high_weekly - low_weekly)
    r3 = high_weekly + 2 * (pivot - low_weekly)
    s3 = low_weekly - 2 * (high_weekly - pivot)
    
    # Weekly volume average (20-period)
    volume_weekly = df_weekly['volume'].values
    vol_ma_weekly = np.full(len(volume_weekly), np.nan)
    for i in range(20, len(volume_weekly)):
        vol_ma_weekly[i] = np.mean(volume_weekly[i-20:i])
    
    # Align weekly data to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    vol_ma_weekly_aligned = align_htf_to_ltf(prices, df_weekly, vol_ma_weekly)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 40  # Need enough data for calculations
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma_weekly_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 6h volume > 1.5x weekly average volume (scaled)
        # Scale weekly volume to 6h: approx 1/(7*4) = 1/28 of weekly volume (7 days * 4 six-hour periods)
        vol_threshold = vol_ma_weekly_aligned[i] / 28.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price drops below S3 OR stoploss hit
            if (close[i] < s3_aligned[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price rises above R3 OR stoploss hit
            if (close[i] > r3_aligned[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 20 bars flat
            if bars_since_entry >= 20:
                # Breakout entries: R3/S3 with volume confirmation
                bull_breakout = close[i] > r3_aligned[i]
                bear_breakout = close[i] < s3_aligned[i]
                
                # Long: break above R3 with volume
                if bull_breakout and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: break below S3 with volume
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