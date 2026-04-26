#!/usr/bin/env python3
"""
6h_WeeklyPivot_Direction_1dTrend_VolumeConfirm_v1
Hypothesis: Weekly pivot points (calculated from prior week OHLC) provide significant support/resistance levels. 
Trade in direction of 1d EMA50 trend when price breaks weekly R1/S1 with volume confirmation (>1.5x 20-bar average). 
Use discrete sizing (0.25) to target ~15-30 trades/year. Works in bull/bear by taking breakouts aligned with daily trend.
Weekly pivots adapt to volatility and provide structural levels that price respects.
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
    
    # Load weekly data ONCE before loop for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivots: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivots to 6h timeframe (wait for weekly close)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Load daily data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of weekly pivot calculation, EMA50 (50), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        trend_val = ema50_1d_aligned[i]
        pivot_val = weekly_pivot_aligned[i]
        r1_val = weekly_r1_aligned[i]
        s1_val = weekly_s1_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Skip if any data not ready
        if (np.isnan(trend_val) or np.isnan(pivot_val) or np.isnan(r1_val) or np.isnan(s1_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price > 1d EMA50 = uptrend, price < 1d EMA50 = downtrend
        is_uptrend = close_val > trend_val
        is_downtrend = close_val < trend_val
        
        # Breakout conditions: price breaks weekly R1/S1
        long_breakout = close_val > r1_val and low_val <= r1_val  # Break above R1
        short_breakout = close_val < s1_val and high_val >= s1_val  # Break below S1
        
        # Entry conditions: Breakout in direction of 1d trend + volume confirmation
        long_entry = long_breakout and is_uptrend and vol_conf
        short_entry = short_breakout and is_downtrend and vol_conf
        
        # Exit conditions: price returns to weekly pivot level (mean reversion to pivot)
        long_exit = close_val < pivot_val
        short_exit = close_val > pivot_val
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_WeeklyPivot_Direction_1dTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0