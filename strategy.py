#!/usr/bin/env python3
"""
6h_WeeklyPivot_Breakout_DailyTrend_Volume
Hypothesis: Use weekly pivot point breakouts (more significant than daily) filtered by daily trend and volume spike. Weekly pivots provide stronger support/resistance levels that are less prone to false breaks. In bull markets, breaks above weekly R1 continue upward; in bear markets, breaks below weekly S1 continue downward. Volume spike confirms institutional interest. Target 15-25 trades/year to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Weekly pivot points from previous week
    # Standard pivot: (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    weekly_high = df_1w['high']
    weekly_low = df_1w['low']
    weekly_close = df_1w['close']
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly levels to 6h timeframe (use previous week's levels)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1.values)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1.values)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for weekly pivot (need 2 weeks), daily EMA, and volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or np.isnan(ema34_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        weekly_r1_level = weekly_r1_aligned[i]
        weekly_s1_level = weekly_s1_aligned[i]
        ema_trend = ema34_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: price closes above weekly R1 + volume spike + uptrend (price > daily EMA34)
            if close[i] > weekly_r1_level and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: price closes below weekly S1 + volume spike + downtrend (price < daily EMA34)
            elif close[i] < weekly_s1_level and vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below weekly S1 or trend turns down
            if close[i] < weekly_s1_level or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above weekly R1 or trend turns up
            if close[i] > weekly_r1_level or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_Breakout_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0