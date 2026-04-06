#!/usr/bin/env python3
"""
6h Weekly Pivot Breakout with Daily Volume Confirmation
Hypothesis: Weekly pivot levels act as strong support/resistance. Breakouts above weekly R1 or below weekly S1 with daily volume confirmation capture institutional interest. Works in bull (buy R1 breakouts) and bear (sell S1 breakdowns). Uses 6h timeframe to reduce noise and trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_v1"
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
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points (standard formula)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    weekly_pivot = np.full(len(weekly_close), np.nan)
    weekly_r1 = np.full(len(weekly_close), np.nan)
    weekly_s1 = np.full(len(weekly_close), np.nan)
    
    for i in range(len(weekly_close)):
        if not (np.isnan(weekly_high[i]) or np.isnan(weekly_low[i]) or np.isnan(weekly_close[i])):
            p = (weekly_high[i] + weekly_low[i] + weekly_close[i]) / 3.0
            weekly_pivot[i] = p
            weekly_r1[i] = 2 * p - weekly_low[i]
            weekly_s1[i] = 2 * p - weekly_high[i]
    
    # Get daily data for volume confirmation
    df_daily = get_htf_data(prices, '1d')
    daily_volume = df_daily['volume'].values
    
    # 20-day average volume
    vol_ma_daily = np.full(len(daily_volume), np.nan)
    for i in range(20, len(daily_volume)):
        vol_ma_daily[i] = np.mean(daily_volume[i-20:i])
    
    # Align weekly pivot levels and daily volume MA to 6h timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    vol_ma_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_daily)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 30  # Need enough data for weekly and daily alignments
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(vol_ma_daily_aligned[i])):
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
            # Exit: price drops below weekly pivot OR stoploss hit (2*ATR)
            if (close[i] < weekly_pivot_aligned[i] if not np.isnan(weekly_pivot_aligned[i]) else False) or \
               close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price rises above weekly pivot OR stoploss hit (2*ATR)
            if (close[i] > weekly_pivot_aligned[i] if not np.isnan(weekly_pivot_aligned[i]) else False) or \
               close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 6 bars flat
            if bars_since_entry >= 6:
                # Breakout entries: R1 breakout for long, S1 breakdown for short
                r1_breakout = close[i] > weekly_r1_aligned[i]
                s1_breakdown = close[i] < weekly_s1_aligned[i]
                
                # Long: breakout above weekly R1 with volume confirmation
                if r1_breakout and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below weekly S1 with volume confirmation
                elif s1_breakdown and volume_filter:
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