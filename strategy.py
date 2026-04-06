#!/usr/bin/env python3
"""
1d Weekly Range Breakout with Volume Filter
Hypothesis: Price breaking above/below weekly high/low on daily timeframe captures
the start of new weekly trends. Weekly trend filter prevents counter-trend trades.
Volume confirms breakout strength. Designed for very low trade frequency (target:
30-100 total over 4 years) to minimize fee drag. Works in bull/bear via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_range_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for range and trend (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly high and low (close-based for simplicity)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Weekly EMA50 for trend filter
    weekly_close = df_weekly['close'].values
    ema50_weekly = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_prev = np.roll(ema50_weekly, 1)
    ema50_weekly_prev[0] = ema50_weekly[0]
    ema50_rising = ema50_weekly > ema50_weekly_prev
    ema50_falling = ema50_weekly < ema50_weekly_prev
    
    # Align weekly data to daily
    weekly_high_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low)
    ema50_rising_aligned = align_htf_to_ltf(prices, df_weekly, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_weekly, ema50_falling)
    
    # Daily data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: 20-day EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup (need weekly data + volume EMA)
    start = 50
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(vol_ema[i]) or np.isnan(ema50_rising_aligned[i]) or 
            np.isnan(ema50_falling_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite weekly level break or stoploss
        if position == 1:  # long position
            # Exit: price breaks below weekly low OR stoploss
            if (close[i] <= weekly_low_aligned[i] or 
                close[i] <= entry_price - 2.5 * (high[i] - low[i] if 'high' in locals() else 0)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above weekly high OR stoploss
            if (close[i] >= weekly_high_aligned[i] or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i] if 'high' in locals() else 0)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: weekly level break + trend + volume
            # Need high/low arrays for stoploss calc
            high = prices['high'].values
            low = prices['low'].values
            
            weekly_high_break = close[i] > weekly_high_aligned[i]
            weekly_low_break = close[i] < weekly_low_aligned[i]
            
            bull_entry = weekly_high_break and ema50_rising_aligned[i] and volume[i] > vol_ema[i] * 1.5
            bear_entry = weekly_low_break and ema50_falling_aligned[i] and volume[i] > vol_ema[i] * 1.5
            
            if bull_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals