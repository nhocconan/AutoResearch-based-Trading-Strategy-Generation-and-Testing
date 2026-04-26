#!/usr/bin/env python3
"""
6h_WeeklyPivot_Direction_1dTrend_VolumeConfirm_v1
Hypothesis: Weekly pivot points (PP, R1, S1) provide key support/resistance levels. 
Price breaking above R1 with 1d uptrend and volume confirmation signals bullish continuation.
Price breaking below S1 with 1d downtrend and volume confirmation signals bearish continuation.
Uses 6h timeframe to capture multi-day moves with tight entry conditions (target: 12-37 trades/year).
Works in both bull/bear markets by following the 1d trend direction.
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
    
    # Load weekly data ONCE before loop for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Previous week's OHLC for weekly pivot calculation (use 1w data)
    prev_week_close = df_1w['close'].shift(1).values
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    
    # Weekly Pivot Points: PP = (H+L+C)/3, R1 = (2*PP) - L, S1 = (2*PP) - H
    weekly_pp = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    weekly_r1 = (2 * weekly_pp) - prev_week_low
    weekly_s1 = (2 * weekly_pp) - prev_week_high
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of EMA34 (34), volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        r1_val = weekly_r1_aligned[i]
        s1_val = weekly_s1_aligned[i]
        pp_val = weekly_pp_aligned[i]
        trend_val = ema34_1d_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Skip if any data not ready
        if (np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(pp_val) or np.isnan(trend_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price > EMA34 = uptrend, price < EMA34 = downtrend
        is_uptrend = close_val > trend_val
        is_downtrend = close_val < trend_val
        
        # Entry conditions: Weekly pivot breakout in direction of trend + volume
        long_condition = (close_val > r1_val) and is_uptrend and vol_conf
        short_condition = (close_val < s1_val) and is_downtrend and vol_conf
        
        # Exit conditions: price crosses weekly pivot point or trend reversal
        long_exit = (position == 1 and (close_val < pp_val or not is_uptrend))
        short_exit = (position == -1 and (close_val > pp_val or not is_downtrend))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_WeeklyPivot_Direction_1dTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0