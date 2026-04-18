#!/usr/bin/env python3
"""
6h_WeeklyPivot_Breakout_Volume
Hypothesis: Trade breakouts from weekly pivot levels (R1/S1) with volume confirmation (>2x average) and trend filter from daily EMA(34). 
Weekly pivots provide strong support/resistance levels that hold across market regimes. Breakouts with volume capture momentum 
after consolidation. Daily EMA(34) filter ensures trades align with intermediate trend, reducing whipsaws in ranging markets. 
Designed for low frequency (target: 20-50 trades/year) to minimize fee drag on 6H timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (HIGHER TIMEFRAME)
    df_weekly = get_htf_data(prices, '1w')
    
    # Get daily data for EMA trend filter
    df_daily = get_htf_data(prices, '1d')
    
    # === WEEKLY PIVOT CALCULATION ===
    # Use prior completed weekly bar
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Shift by 1 to get previous week's data (completed bar)
    prev_weekly_high = np.roll(weekly_high, 1)
    prev_weekly_low = np.roll(weekly_low, 1)
    prev_weekly_close = np.roll(weekly_close, 1)
    # First bar: use same week's data (no look-ahead)
    prev_weekly_high[0] = weekly_high[0]
    prev_weekly_low[0] = weekly_low[0]
    prev_weekly_close[0] = weekly_close[0]
    
    # Calculate pivot points from previous week
    pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    r1 = 2 * pivot - prev_weekly_low
    s1 = 2 * pivot - prev_weekly_high
    r2 = pivot + (prev_weekly_high - prev_weekly_low)
    s2 = pivot - (prev_weekly_high - prev_weekly_low)
    
    # === DAILY EMA TREND FILTER ===
    daily_close = df_daily['close'].values
    ema_period = 34
    ema_daily = np.full_like(daily_close, np.nan)
    if len(daily_close) >= ema_period:
        # Calculate EMA with proper initialization
        ema_daily[ema_period-1] = np.mean(daily_close[:ema_period])
        for i in range(ema_period, len(daily_close)):
            ema_daily[i] = (daily_close[i] * 2 / (ema_period + 1)) + (ema_daily[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # === VOLUME FILTER ===
    vol_period = 20
    vol_ma = np.full_like(volume, np.nan)
    for i in range(vol_period, len(volume)):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # ALIGN HIGHER TIMEFRAME INDICATORS TO 6H CHART
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Initialize signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after all indicators are valid
    start_idx = max(50, vol_period, ema_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_daily_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: volume > 2x average
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # LONG: Break above R1 with volume and above daily EMA
            if close[i] > r1_aligned[i] and vol_confirm and close[i] > ema_daily_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 with volume and below daily EMA
            elif close[i] < s1_aligned[i] and vol_confirm and close[i] < ema_daily_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # LONG EXIT: Price closes below S1 (reversal) or below daily EMA
            if close[i] < s1_aligned[i] or close[i] < ema_daily_aligned[i]:
                signals[i] = -0.25  # Reverse to short
                position = -1
            else:
                signals[i] = 0.25  # Hold long
        
        elif position == -1:
            # SHORT EXIT: Price closes above R1 (reversal) or above daily EMA
            if close[i] > r1_aligned[i] or close[i] > ema_daily_aligned[i]:
                signals[i] = 0.25  # Reverse to long
                position = 1
            else:
                signals[i] = -0.25  # Hold short
    
    return signals

name = "6h_WeeklyPivot_Breakout_Volume"
timeframe = "6h"
leverage = 1.0