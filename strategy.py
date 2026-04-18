#!/usr/bin/env python3
"""
6h_WeeklyPivot_DailyTrend_Filter
Hypothesis: Trade weekly pivot breakouts on 6h with daily trend filter. 
In both bull and bear markets, price tends to respect weekly pivot levels (S1/S2/R1/R2). 
Breakouts above R1 or below S1 with daily EMA50 alignment indicate strong momentum. 
Volume confirmation filters false breakouts. 
Targets 15-30 trades/year via strict pivot breakout criteria + volume + trend filter.
Works in bull by following upward breaks, in bear by shorting downward breaks.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using typical price)
    typical_price_weekly = (df_weekly['high'] + df_weekly['low'] + df_weekly['close']) / 3
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Pivot point calculation
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    
    # Get daily data for trend filter (EMA50)
    df_daily = get_htf_data(prices, '1d')
    close_daily = df_daily['close'].values
    
    # Calculate EMA50 on daily
    ema50_daily = np.full_like(close_daily, np.nan)
    if len(close_daily) >= 50:
        ema50_daily[49] = np.mean(close_daily[:50])
        for i in range(50, len(close_daily)):
            ema50_daily[i] = (close_daily[i] * 2 / (50 + 1)) + (ema50_daily[i-1] * (48 / (50 + 1)))
    
    # Align daily EMA50 to 6h
    ema50_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, vol_period)  # Need EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 + daily close > EMA50 + volume
            if close[i] > r1_aligned[i] and close[i] > ema50_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + daily close < EMA50 + volume
            elif close[i] < s1_aligned[i] and close[i] < ema50_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls back below pivot or daily close < EMA50
            if close[i] < pivot_aligned[i] or close[i] < ema50_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above pivot or daily close > EMA50
            if close[i] > pivot_aligned[i] or close[i] > ema50_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_DailyTrend_Filter"
timeframe = "6h"
leverage = 1.0