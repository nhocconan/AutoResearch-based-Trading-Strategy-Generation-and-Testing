#!/usr/bin/env python3
"""
12h_1d_WeeklyPivot_TrendBreakout_Volume
Hypothesis: Trade breakouts above/below weekly Camarilla P1/S1 levels in direction of daily EMA(34) trend, confirmed by volume >1.5x 20-period average. Uses daily trend filter to avoid counter-trend trades. Position size 0.25 targeting ~20 trades/year to minimize fee drag. Works in bull/bear by trading breakouts with trend alignment and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly Pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1d calculations (previous week's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous week's OHLC (completed week) - using 5-day approximation for weekly
    # We'll use 5-period lookback to approximate weekly data
    if len(high_1d) >= 5:
        # Calculate weekly high/low/close from last 5 days
        week_high = np.full_like(high_1d, np.nan)
        week_low = np.full_like(low_1d, np.nan)
        week_close = np.full_like(close_1d, np.nan)
        
        for i in range(5, len(high_1d)):
            week_high[i] = np.max(high_1d[i-5:i])
            week_low[i] = np.min(low_1d[i-5:i])
            week_close[i] = close_1d[i-1]  # Previous day's close
        
        # Previous week's OHLC
        prev_week_high = np.roll(week_high, 5)
        prev_week_low = np.roll(week_low, 5)
        prev_week_close = np.roll(week_close, 5)
        
        # Initialize first values
        prev_week_high[:5] = week_high[0] if not np.isnan(week_high[0]) else high_1d[0]
        prev_week_low[:5] = week_low[0] if not np.isnan(week_low[0]) else low_1d[0]
        prev_week_close[:5] = week_close[0] if not np.isnan(week_close[0]) else close_1d[0]
    else:
        prev_week_high = np.full_like(high_1d, np.nan)
        prev_week_low = np.full_like(high_1d, np.nan)
        prev_week_close = np.full_like(high_1d, np.nan)
    
    # Weekly Camarilla P1 and S1 levels (based on previous week)
    P1 = np.full_like(high_1d, np.nan)
    S1 = np.full_like(low_1d, np.nan)
    
    for i in range(5, len(high_1d)):
        if not (np.isnan(prev_week_high[i]) or np.isnan(prev_week_low[i]) or np.isnan(prev_week_close[i])):
            range_ = prev_week_high[i] - prev_week_low[i]
            P1[i] = prev_week_close[i] + range_ * 1.1 / 12
            S1[i] = prev_week_close[i] - range_ * 1.1 / 12
    
    # 1w EMA trend filter (34-period)
    close_1w = df_1w['close'].values
    ema_period = 34
    ema_1w = np.full_like(close_1w, np.nan)
    
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 / (ema_period + 1)) + (ema_1w[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Align weekly Pivot levels to 12h timeframe
    P1_aligned = align_htf_to_ltf(prices, df_1d, P1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Align weekly EMA to 12h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, vol_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(P1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above P1 with volume and above weekly EMA
            if close[i] > P1_aligned[i] and vol_confirm and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and below weekly EMA
            elif close[i] < S1_aligned[i] and vol_confirm and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below S1 (reverse signal) or below weekly EMA
            if close[i] < S1_aligned[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above P1 (reverse signal) or above weekly EMA
            if close[i] > P1_aligned[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_WeeklyPivot_TrendBreakout_Volume"
timeframe = "12h"
leverage = 1.0