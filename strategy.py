#!/usr/bin/env python3
"""
1d_WeeklyPivot_HighLow_Breakout_TrendFilter
Hypothesis: On daily timeframe, go long when price breaks above weekly pivot high in uptrend (price > weekly EMA20), short when price breaks below weekly pivot low in downtrend (price < weekly EMA20), with volume confirmation (volume > 1.5x 20-day average). Designed for 1d timeframe to capture multi-day trends while minimizing trades and fee drag. Uses weekly pivot points for structure and EMA20 for trend filter.
"""

name = "1d_WeeklyPivot_HighLow_Breakout_TrendFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (based on prior week)
    prev_weekly_high = df_weekly['high'].shift(1).values
    prev_weekly_low = df_weekly['low'].shift(1).values
    prev_weekly_close = df_weekly['close'].shift(1).values
    
    valid_idx = ~np.isnan(prev_weekly_high) & ~np.isnan(prev_weekly_low) & ~np.isnan(prev_weekly_close)
    weekly_pivot_high = np.full_like(prev_weekly_close, np.nan)
    weekly_pivot_low = np.full_like(prev_weekly_close, np.nan)
    
    weekly_pivot_high[valid_idx] = prev_weekly_high[valid_idx]
    weekly_pivot_low[valid_idx] = prev_weekly_low[valid_idx]
    
    # Align weekly pivot levels to daily timeframe
    weekly_pivot_high_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot_high)
    weekly_pivot_low_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot_low)
    
    # Get weekly EMA20 for trend filter
    ema_20_weekly = pd.Series(df_weekly['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_20_weekly)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    cooldown = 0  # cooldown counter to prevent immediate re-entry
    
    for i in range(30, n):
        # Decrease cooldown if active
        if cooldown > 0:
            cooldown -= 1
        
        if position == 0 and cooldown == 0:
            # LONG: Price breaks above weekly pivot high with volume confirmation in uptrend
            if weekly_pivot_high_aligned[i] > 0 and not np.isnan(weekly_pivot_high_aligned[i]) and \
               high[i] > weekly_pivot_high_aligned[i] and volume_confirmed[i] and \
               close[i] > ema_20_weekly_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly pivot low with volume confirmation in downtrend
            elif weekly_pivot_low_aligned[i] > 0 and not np.isnan(weekly_pivot_low_aligned[i]) and \
                 low[i] < weekly_pivot_low_aligned[i] and volume_confirmed[i] and \
                 close[i] < ema_20_weekly_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back below weekly pivot low or trend weakens
            if weekly_pivot_low_aligned[i] > 0 and not np.isnan(weekly_pivot_low_aligned[i]) and \
               (low[i] < weekly_pivot_low_aligned[i] or close[i] < ema_20_weekly_aligned[i]):
                signals[i] = 0.0
                position = 0
                cooldown = 5  # 5-day cooldown after exit
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back above weekly pivot high or trend weakens
            if weekly_pivot_high_aligned[i] > 0 and not np.isnan(weekly_pivot_high_aligned[i]) and \
               (high[i] > weekly_pivot_high_aligned[i] or close[i] > ema_20_weekly_aligned[i]):
                signals[i] = 0.0
                position = 0
                cooldown = 5  # 5-day cooldown after exit
            else:
                signals[i] = -0.25
    
    return signals