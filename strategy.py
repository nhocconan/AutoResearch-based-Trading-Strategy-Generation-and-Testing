#!/usr/bin/env python3
"""
6h_WeeklyPivot_MonthlyTrend_VolumeBreakout_v1
Hypothesis: Trade long when price breaks above weekly R1 with monthly uptrend (price > monthly SMA50) and volume > 1.5x 24-period average; short when price breaks below weekly S1 with monthly downtrend (price < monthly SMA50) and volume confirmation. Uses weekly pivots for structure and monthly trend filter to avoid counter-trend trades. Targets 15-30 trades/year via strict breakout conditions. Works in bull by following uptrend breaks and in bear by shorting downtrend breaks. Volume confirmation reduces false breakouts.
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
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly pivot points (standard calculation)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate pivots: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivots to 6h
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    
    # Get monthly data for trend filter (SMA 50)
    df_monthly = get_htf_data(prices, '1M')
    
    # Monthly SMA 50
    monthly_close = df_monthly['close'].values
    monthly_sma50 = np.full_like(monthly_close, np.nan)
    
    if len(monthly_close) >= 50:
        for i in range(50, len(monthly_close)):
            monthly_sma50[i] = np.mean(monthly_close[i-50:i])
    
    # Align monthly SMA50 to 6h
    monthly_sma50_aligned = align_htf_to_ltf(prices, df_monthly, monthly_sma50)
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, vol_period)  # monthly SMA needs 50, vol MA needs 24
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(monthly_sma50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above weekly R1 + monthly uptrend (price > monthly SMA50) + volume
            if close[i] > weekly_r1_aligned[i] and close[i] > monthly_sma50_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 + monthly downtrend (price < monthly SMA50) + volume
            elif close[i] < weekly_s1_aligned[i] and close[i] < monthly_sma50_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below weekly S1 or monthly trend turns down (price < monthly SMA50)
            if close[i] < weekly_s1_aligned[i] or close[i] < monthly_sma50_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above weekly R1 or monthly trend turns up (price > monthly SMA50)
            if close[i] > weekly_r1_aligned[i] or close[i] > monthly_sma50_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_MonthlyTrend_VolumeBreakout_v1"
timeframe = "6h"
leverage = 1.0