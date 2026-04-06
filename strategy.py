#!/usr/bin/env python3
"""
6h Weekly Pivot + Donchian(20) Breakout
Hypothesis: Use weekly pivot points (from 1w data) to establish key support/resistance zones. Trade Donchian(20) breakouts in the direction of weekly pivot bias (above weekly pivot = bullish bias, below = bearish). Add volume confirmation from 1d data to ensure institutional participation. Works in bull markets (buy breakouts above weekly pivot) and bear markets (sell breakdowns below weekly pivot). Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_donchian20_v1"
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
    
    # Get weekly data for pivot points (key levels)
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P-L, S1 = 2*P-H, etc.
    weekly_pivot = np.full(len(weekly_close), np.nan)
    weekly_r1 = np.full(len(weekly_close), np.nan)
    weekly_s1 = np.full(len(weekly_close), np.nan)
    weekly_r2 = np.full(len(weekly_close), np.nan)
    weekly_s2 = np.full(len(weekly_close), np.nan)
    
    for i in range(len(weekly_close)):
        if not (np.isnan(weekly_high[i]) or np.isnan(weekly_low[i]) or np.isnan(weekly_close[i])):
            pivot = (weekly_high[i] + weekly_low[i] + weekly_close[i]) / 3.0
            weekly_pivot[i] = pivot
            weekly_r1[i] = 2 * pivot - weekly_low[i]
            weekly_s1[i] = 2 * pivot - weekly_high[i]
            weekly_r2[i] = pivot + (weekly_high[i] - weekly_low[i])
            weekly_s2[i] = pivot - (weekly_high[i] - weekly_low[i])
    
    # Align weekly pivot levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    r1_6h = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    s1_6h = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    r2_6h = align_htf_to_ltf(prices, df_weekly, weekly_r2)
    s2_6h = align_htf_to_ltf(prices, df_weekly, weekly_s2)
    
    # Get daily data for volume confirmation
    df_daily = get_htf_data(prices, '1d')
    volume_daily = df_daily['volume'].values
    
    # 20-period average volume on daily
    vol_ma_daily = np.full(len(volume_daily), np.nan)
    for i in range(20, len(volume_daily)):
        vol_ma_daily[i] = np.mean(volume_daily[i-20:i])
    
    # Align daily volume MA to 6h timeframe
    vol_ma_daily_6h = align_htf_to_ltf(prices, df_daily, vol_ma_daily)
    
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
        if (np.isnan(atr[i]) or np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or 
            np.isnan(s1_6h[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(vol_ma_daily_6h[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 6h volume > 1.5x daily average volume (scaled)
        # Scale daily volume to 6h: approx 1/4 of daily volume (since 4x 6h in 1d)
        vol_threshold = vol_ma_daily_6h[i] / 4.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR below weekly S1 (strong support break)
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lower[i] or
                close[i] < s1_6h[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR above weekly R1 (strong resistance break)
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > upper[i] or
                close[i] > r1_6h[i] or
                close[i] > entry_price + 2.0 * atr[i]):
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
                # Breakout entries: upper/lower Donchian with weekly pivot bias
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                
                # Weekly pivot bias: above pivot = bullish, below pivot = bearish
                bullish_bias = close[i] > pivot_6h[i]
                bearish_bias = close[i] < pivot_6h[i]
                
                # Long: breakout above upper with bullish bias + volume
                if bull_breakout and bullish_bias and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below lower with bearish bias + volume
                elif bear_breakout and bearish_bias and volume_filter:
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